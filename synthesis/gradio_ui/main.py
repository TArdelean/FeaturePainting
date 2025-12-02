import pickle
from pathlib import Path

import gradio as gr
import gradio_mycomponent
import torch
import torchvision
from gradio import Brush
from torchvision.transforms import v2

import dnnlib
import my_utils
from my_utils import edm_sampler_guided, stacked_random_latents, PostProcessor, ddim_invert_fpi
from torch_utils import misc
from training.my_networks import EDMPrecondSPE

# gradio_colors = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 255], [0, 255, 0]]
gradio_colors = [[0, 0, 0], [0, 158, 115], [213, 94, 0], [0, 114, 178], [240, 228, 66], [204, 121, 167]]
inversion_steps_edit = 200


def load_network(device, exp_path):
    weight_paths = sorted(exp_path.glob('*.pkl'))
    network_pkl = str(weight_paths[-1])
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=True) as f:
        net = pickle.load(f)['ema'].to(device)
    new_net = EDMPrecondSPE(*net.init_args, **net.init_kwargs).to(device)
    misc.copy_params_and_buffers(net, new_net, require_all=True)
    net = new_net

    net.round_sigma = lambda sigma: torch.as_tensor(sigma)
    net.sigma_min = 0.002
    net.sigma_max = 80
    return net.eval()


def single_generation(net, class_labels, guidance_scale):
    batch_size = class_labels.shape[0]
    h, w = class_labels.shape[-2:]
    rnd, latents = stacked_random_latents(net, batch_size, class_labels.device, h=h, w=w)
    images = edm_sampler_guided(net, latents, class_labels, randn_like=rnd.randn_like, guidance=guidance_scale)
    return images


def multi_generation(net, class_labels, guidance_scale):
    total_h, total_w = class_labels.shape[-2:]
    assert total_h >= 64 and total_w >= 64 and total_h % 64 == 0 and total_w % 64 == 0
    rnd, latents = stacked_random_latents(net, 1, class_labels.device, h=total_h, w=total_w)
    latents = my_utils.noise_uniformization_technique(latents, ih=64, iw=64)
    return my_utils.multi_diffusion(net, latents, class_labels, rnd.randn_like, guidance=guidance_scale,
                                    num_steps=20, overlap=32, randomize=15)


def class_labels_from_sketch(sketch, colors, device, hw):
    colors = torch.tensor(colors, dtype=torch.float32, device=device)
    sketch = torch.tensor(sketch[:, :, :3], dtype=torch.float32, device=device).permute(2, 0, 1)
    if hw[0] != sketch.shape[-2] or hw[1] != sketch.shape[-1]:
        sketch = torch.nn.functional.interpolate(sketch[None], hw, mode='nearest', antialias=False)[0]
    idxs = torch.abs(sketch[None, :, :, :] - colors[:, :, None, None]).mean(dim=1).argmin(dim=0)
    class_labels = torch.nn.functional.one_hot(idxs, num_classes=len(colors)).permute(2, 0, 1)[None].float()
    return class_labels


def run_model(net, class_labels, hw, guidance_scale=4.0):
    if hw[0] * hw[1] <= 128 * 128:
        return single_generation(net, class_labels, guidance_scale)
    else:
        return multi_generation(net, class_labels, guidance_scale)


def make_edit(net, original_noise, original_dirs, class_labels, hw, guidance_scale=4.0, num_inv_steps=250, mixing=0.5):
    if hw == (net.img_resolution, net.img_resolution):
        return my_utils.noise_mixing_edm_edit(net, original_noise, original_dirs, class_labels, num_inv_steps,
                                              guidance=guidance_scale, mixing_alpha=mixing)
    else:
        raise Exception("Not implemented")

def rgb_to_hex(rgb):
    return "#" + '%02x%02x%02x' % tuple(rgb)

def load_weights_options(weights_path):
    return sorted([wp.name for wp in weights_path.iterdir()], reverse=True)

def main():
    device = torch.device('cuda:0')
    weights_dir_edm = Path('synthesis-runs')
    weights_dir = weights_dir_edm
    options = load_weights_options(weights_dir)
    colors_hex = [rgb_to_hex(c) for c in gradio_colors]
    post = PostProcessor().load_sd(device)
    state = {"current_weights": options[0], "net": load_network(device, weights_dir / options[0])}

    def update_weights(weights_option):
        if weights_option != state["current_weights"]:
            state["net"] = load_network(device, weights_dir / weights_option)
            state["current_weights"] = weights_option

    def generate(weights_option, guidance_scale, height, width, sketch, mixing):
        sketch = sketch['composite']
        msg = f"Loaded {weights_option}, sketch size {sketch.shape} to {height} x {width}"
        update_weights(weights_option)
        n_classes = state["net"].label_dim
        class_labels = class_labels_from_sketch(sketch, gradio_colors[:n_classes], device, hw=(height, width))
        if state.get('original_noise', None) is None:
            result = run_model(state["net"], class_labels, (height, width), guidance_scale)
        else:
            result = make_edit(state["net"], state['original_noise'], state['original_dirs'],
                               class_labels, (height, width), guidance_scale,
                               num_inv_steps=inversion_steps_edit, mixing=mixing)
            msg += "\nEditing mode running"
        result_np = post.to_numpy(result)[0]
        return msg, result_np

    @torch.no_grad()
    def load_img_to_edit(weights_option, image_path, inv_guidance=0.0, num_inv_steps=inversion_steps_edit, fpi_iter=4):
        msg = f"Loaded {str(image_path)}\n"
        update_weights(weights_option)
        to_i_img = torchvision.io.read_image(image_path).to(device)
        # total_h, total_w = to_i_img.shape[-2:]
        total_h, total_w = 512, 512
        to_i_img = torch.nn.functional.interpolate(to_i_img[None] / 255.0, (total_h, total_w))
        to_i_img = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(to_i_img)

        to_invert = post.encode(to_i_img)

        pure_labels = my_utils.get_pure_labels(total_h // 8, total_w // 8, device, state["net"].label_dim)
        inv_noise, last_t, dirs = ddim_invert_fpi(state["net"], to_invert, pure_labels, guidance=inv_guidance,
                                                  num_steps=num_inv_steps, fpi_iter=fpi_iter)
        inv_noise /= last_t
        state['original_noise'] = inv_noise
        state['original_dirs'] = dirs
        result = edm_sampler_guided(state["net"], inv_noise, pure_labels, guidance=inv_guidance,
                                    num_steps=num_inv_steps, only_euler=True)
        msg += f"L1 to invert (noise space): {(result - to_invert).abs().mean().item()}\n"
        msg += f"L1 to original (normalized pixel space): {(post.decode(result) - to_i_img).abs().mean().item()}\n"
        result_np = post.to_numpy(result)[0]
        return msg, result_np

    def clear_img_to_edit(image):
        del state['original_noise']
        del state['original_dirs']
        return "Cleared", None

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                network_selector = gr.Dropdown(options, value=options[0], label="Weights",
                                               info="Which network weights to load", interactive=True)
                guidance_slider = gr.Slider(minimum=1.0, maximum=10.0, value=5.0, label="Guidance scale")
                mixing_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.3, label="Mixing alpha")
                height = gr.Number(value=64, minimum=64, maximum=512, label="Height")
                width = gr.Number(value=64, minimum=64, maximum=512, label="Width")
                sketch = gradio_mycomponent.MyComponent(sources=['upload'], label="Conditioning map",
                                                        brush=Brush(colors=colors_hex[1:], color_mode='fixed'),
                                                        crop_size="8:8",
                                                        image_mode="RGB")
                generate_btn = gr.Button("Generate")
            with gr.Column():
                out_txt = gr.Textbox(label="Output log")
                out_img = gr.Image()
        with gr.Row():
            with gr.Column():
                original_img = gr.Image(type="filepath", label="Image to edit", sources=['upload', 'clipboard'])
                edit_log_txt = gr.Textbox("No image loaded", label="Info")
                with gr.Column():
                    load_img_btn = gr.Button("Invert image")
                with gr.Column():
                    clear_img_btn = gr.Button("Clear image")
            with gr.Column():
                reconstructed_img = gr.Image(label="Reconstructed image (diff inv)")
            load_img_btn.click(fn=load_img_to_edit, inputs=[network_selector, original_img],
                               outputs=[edit_log_txt, reconstructed_img])
            clear_img_btn.click(fn=clear_img_to_edit, inputs=[original_img], outputs=[edit_log_txt, original_img])

        generate_btn.click(fn=generate,
                           inputs=[network_selector, guidance_slider, height, width, sketch, mixing_slider],
                           outputs=[out_txt, out_img], api_name="generate")

    demo.launch(share=False)


if __name__ == '__main__':
    main()
