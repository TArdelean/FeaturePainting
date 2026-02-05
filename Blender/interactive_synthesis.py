import json
import sys
import cv2
import traceback
from pathlib import Path
import torch
import numpy as np
from PIL import Image

project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir / "synthesis"))
import my_utils
from my_utils import PostProcessor, noise_mixing_edm_edit, get_pure_labels
from gradio_ui.main import load_network, class_labels_from_sketch, gradio_colors


weights_dir = project_dir / 'synthesis-runs'
device = torch.device("cuda:0")
global_state = {'net': None, 'loaded_weight': "", 'post': PostProcessor().load_sd(device),
                'stem': "", "latent": None, "dirs": None, "gn_stats": None}


def cached_network(weight_name):
    if weight_name != global_state['loaded_weight']:
        net = load_network(device, weights_dir / weight_name)
        print(f"[INFO]: Loaded network {weight_name}", flush=True)
        global_state['loaded_weight'] = weight_name
        global_state['net'] = net
    else:
        net = global_state['net']
    return net


def cached_trajectory(marked_path):
    if marked_path.stem != global_state["stem"]:
        latent = torch.load(marked_path.parent.parent / "noise" / f"{marked_path.stem}.pt", map_location=device, weights_only=True)
        dirs = torch.load(marked_path.parent.parent / "dirs" / f"{marked_path.stem}.pt", map_location=device, weights_only=True)
        gn_stats = torch.load(marked_path.parent.parent / "gn_stats" / f"{marked_path.stem}.pt", map_location=device, weights_only=True)
        global_state['latent'] = latent
        global_state['dirs'] = dirs
        global_state['gn_stats'] = gn_stats
        global_state["stem"] = marked_path.stem
    else:
        latent = global_state['latent']
        dirs = global_state['dirs']
        gn_stats = global_state['gn_stats']
    return latent, dirs, gn_stats


def compute_sketch(marked_path):
    original_path = marked_path.parent.parent / "synth" / marked_path.name
    marked_im = np.array(Image.open(marked_path).convert('RGB'))
    original_im = np.array(Image.open(original_path).convert('RGB'))

    unchanged = np.mean(np.abs(marked_im.astype(np.float32) - original_im.astype(np.float32)), axis=-1, keepdims=True)
    unchanged = np.broadcast_to(unchanged < 10, marked_im.shape)
    sketch = np.where(unchanged, np.zeros_like(marked_im), marked_im)
    return sketch


def make_task(msg):
    return json.loads(msg)


def generate_image(task):
    marked_path = Path(task["filepath"])
    # id_to_weight_map[marked_path.stem] = global_state['loaded_weight']
    save_dir = marked_path.parent.parent

    net = global_state['net']
    post = global_state['post']
    gn_stats = {}
    my_utils.setup_post_processor_with_gn_stats(post, gn_stats, use_stats=False)

    h, w = task.get('h', 512) // 8, task.get('w', 512) // 8
    class_labels = get_pure_labels(h, w, device, net.label_dim)
    rnd, latents = my_utils.stacked_random_latents(net, 1, class_labels.device, h=h, w=w)
    result, directions = my_utils.euler_sampler_trajectory(net, latents, class_labels, num_steps=42,
                                                           guidance=task["guidance"])
    directions = torch.cat(directions, dim=0)
    # result = single_generation(net, class_labels, guidance_scale=task["guidance"])
    result_np = post.to_numpy(result)[0]
    Image.fromarray(result_np).save(save_dir / "synth" / marked_path.name)
    Image.fromarray(result_np).save(save_dir / "original" / marked_path.name)

    torch.save(latents, save_dir / "noise" / f"{marked_path.stem}.pt")
    torch.save(directions, save_dir / "dirs" / f"{marked_path.stem}.pt")
    torch.save(gn_stats, save_dir / "gn_stats" / f"{marked_path.stem}.pt")
    global_state["stem"] = ""  # Force reloading for the next edit

    return {"task": task, "result": str(save_dir / "synth" / marked_path.name)}


def get_square_bounding_box(arr, min_size, mof=32):
    assert arr.ndim == 2, "Input must be a 2D array"
    # Find coordinates of True elements
    true_y, true_x = np.where(arr)

    if len(true_x) == 0 or len(true_y) == 0:
        return 0, min_size, 0, min_size

    # Find bounding box of True region
    x_min, x_max = true_x.min(), true_x.max()
    y_min, y_max = true_y.min(), true_y.max()

    # Current box size
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    size = ((max(width, height, min_size) - 1) // mof + 1) * mof

    # Center of the original bounding box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Calculate new bounds
    half_size = size // 2
    x_start = center_x - half_size
    y_start = center_y - half_size

    # Clip to array bounds
    x_start = max(0, x_start)
    y_start = max(0, y_start)

    # Adjust for odd sizes
    x_end = x_start + size
    y_end = y_start + size

    x_end = min(arr.shape[1], x_end)
    y_end = min(arr.shape[0], y_end)

    # If we clipped, adjust start to maintain size if possible
    x_start = max(0, x_end - size)
    y_start = max(0, y_end - size)

    return y_start, y_end, x_start, x_end


def extract_bounds(sketch):
    altered = sketch.astype(np.int32).sum(axis=-1) > 0.5
    y_start, y_end, x_start, x_end = get_square_bounding_box(altered, min_size=448)
    return y_start // 8 * 8, y_end // 8 * 8, x_start // 8 * 8, x_end // 8 * 8


def blend_in(big_frame, bounds, small_frame, off_sf=(64, 64)):
    # big_frame[bounds[0]:bounds[1], bounds[2]:bounds[3], :] = small_frame
    # Linear blending
    k0, k1 = small_frame.shape[0], small_frame.shape[1]
    row_indices = np.arange(k0)
    col_indices = np.arange(k1)
    relu = lambda x: x * (x > 0)

    # Generate the 2D grid (row-column order)
    grid_i, grid_j = np.meshgrid(row_indices, col_indices, indexing='ij')
    gi_rew = np.maximum(relu(off_sf[0] - grid_i), relu(grid_i - k0 + off_sf[0] + 1))
    gj_rew = np.maximum(relu(off_sf[1] - grid_j), relu(grid_j - k1 + off_sf[1] + 1))
    linear_w = 1 - np.maximum(gi_rew / (off_sf[0] + 1), gj_rew / (off_sf[1] + 1))
    linear_w = linear_w[:, :, None]

    bfc = big_frame[bounds[0]:bounds[1], bounds[2]:bounds[3], :].astype(np.float32)
    bfc *= (1 - linear_w)
    bfc += linear_w * small_frame
    big_frame[bounds[0]:bounds[1], bounds[2]:bounds[3], :] = bfc.astype(np.uint8)
    # return (linear_w * 255).astype(np.uint8)
    return big_frame

def blend_in_masked(big_frame, bounds, small_frame, sketch, off_sf=25):
    """
    Blends in the small frame into the big frame according to the sketch

    Args:
        big_frame: np.array, shape H x W x 3, np.uint8
        bounds: The position of the small_frame into the big frame is given by bounds[0]:bounds[1], bounds[2]:bounds[3].
        small_frame: np.array, shape h x w x 3, np.uint8 (bounds[1] - bounds[0] == h, and bounds[3] - bounds[2] == w)
        sketch: small_frame sized sketch
        off_sf: dilation of the sketch for a smooth blending

    Returns:
        np.array, shape H x W x 3, np.uint8
    """
    y0, y1, x0, x1 = bounds
    roi = big_frame[y0:y1, x0:x1].astype(np.float32).copy()
    small = small_frame.astype(np.float32)

    mask = (sketch.astype(np.int32).sum(axis=-1) > 0.5).astype(np.uint8)
    # Dilate mask for smoother blending
    if off_sf > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (off_sf, off_sf))
        mask = cv2.dilate(mask, kernel)

    dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    dist = np.clip(dist / off_sf, 0, 1)

    alpha = 1.0 - dist
    alpha = alpha[..., None]  # shape (h, w, 1)

    # Blend
    blended = alpha * small + (1 - alpha) * roi
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    big_frame[y0:y1, x0:x1] = blended
    return big_frame


def preload_weights(task):
    weight_id = task.get("weight", None)
    cached_network(weight_id)
    # cached_trajectory(marked_path)
    return {"task": task, "result": f"Preloaded {weight_id}"}


def edit_image(task):
    marked_path = Path(task["filepath"])

    net = global_state['net']
    post = global_state['post']
    sketch = compute_sketch(marked_path)
    bounds = extract_bounds(sketch)
    sketch = sketch[bounds[0]:bounds[1], bounds[2]:bounds[3], :]

    print("Sketch shape", sketch.shape)
    class_labels = class_labels_from_sketch(sketch, gradio_colors[:net.label_dim], device,
                                            hw=(sketch.shape[0] // 8, sketch.shape[1] // 8))
    latent, dirs, gn_stats = cached_trajectory(marked_path)
    my_utils.setup_post_processor_with_gn_stats(post, gn_stats, use_stats=True)

    crop = np.index_exp[:, :, bounds[0] // 8: bounds[1] // 8, bounds[2] // 8: bounds[3] // 8]
    result = noise_mixing_edm_edit(net, latent[crop], torch.flip(dirs[crop], [0]), class_labels,
                                   num_steps=len(dirs), guidance=task["guidance"], enforce_om=True, mixing_alpha=1.0)

    result_np = post.to_numpy(result)[0]
    synth_path = marked_path.parent.parent / "synth" / marked_path.name
    full_np = np.array(Image.open(synth_path).convert('RGB'))
    full_np = blend_in_masked(full_np, bounds, result_np, sketch)
    Image.fromarray(full_np).save(marked_path.parent.parent / "synth" / marked_path.name)
    return {"task": task, "result": str(marked_path.parent.parent / "synth" / marked_path.name)}


def dispatch_task(task):
    try:
        if task['method'] == 'generate_image':
            return generate_image(task)
        elif task['method'] == 'edit_image':
            return edit_image(task)
        elif task['method'] == 'preload_weights':
            return preload_weights(task)
        else:
            raise Exception(f"Undefined method {task['method']}")
    except Exception as e:
        error_msg = traceback.format_exc()
        return {"task": task, "result": "ERROR", "reason": str(e), "stack_track": error_msg}


def process_command(msg):
    task = make_task(msg)
    print(f"[INFO]: Working on {task['method']} with args {str(task)}", flush=True)
    if task['method'] == "exit":
        print("[INFO]: Received exit signal", flush=True)
        return False
    print(f"[MSG]: {json.dumps(dispatch_task(task))}", flush=True)
    print(f"[INFO]: {task['method']} complete. DONE", flush=True)
    return True


print("[INFO]: Ready to receive commands.", flush=True)

for line in sys.stdin:
    command = line.strip()
    if not process_command(command):
        break
