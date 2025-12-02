import torch
import torch.nn.functional as F
import numpy as np

from training.my_networks import SDLatentVAE


# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, tensor):
        return self.randn(tensor.shape, dtype=tensor.dtype, layout=tensor.layout, device=tensor.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def stacked_random_latents(net, batch_size, device, h=None, w=None):
    batch_seeds = np.random.randint(0, 19999, size=batch_size)
    rnd = StackedRandomGenerator(device, batch_seeds)
    if h is None:
        h = net.img_resolution
    if w is None:
        w = net.img_resolution
    latents = rnd.randn((batch_size, net.img_channels, h, w), device=device)
    return rnd, latents


def get_pure_labels(h, w, device, num_classes):
    pure_labels = torch.zeros((h, w), dtype=torch.long, device=device)
    pure_labels = torch.nn.functional.one_hot(pure_labels, num_classes=num_classes).permute(2, 0, 1)[None].float()
    return pure_labels


def blur(image, kernel_size=7, sigma=None, rescale=False, padding='circular'):
    if sigma is None:
        sigma = kernel_size / 4
    shape = image.shape
    im_b = image[(None,) * (4 - len(shape))]

    if rescale:
        im_b = torch.cat([torch.ones_like(im_b[:, :1]), im_b], dim=1)
    kernel_1d = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=im_b.device)
    kernel_1d = torch.exp(-(kernel_1d ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    channels = im_b.shape[1]
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)

    im_b = torch.nn.functional.pad(im_b, (kernel_size // 2,) * 4, mode=padding, value=0)
    blurred = torch.nn.functional.conv2d(im_b, kernel, groups=channels)
    if rescale:
        blurred = blurred[:, 1:] / blurred[:, :1]
    return blurred.reshape(shape)


def get_wrapped_window(tensor, pos_i, pos_j, size_i, size_j):
    H, W = tensor.shape[-2:]
    x_start = pos_i % H
    x_end = (pos_i + size_i) % H

    if x_end >= x_start:
        rows = tensor[..., x_start:x_end, :]
    else:  # If it wraps around, we concatenate the two slices
        rows = torch.cat([tensor[..., x_start:, :], tensor[..., :x_end, :]], dim=-2)

    y_start = pos_j % W
    y_end = (pos_j + size_j) % W

    if y_end >= y_start:
        cols = rows[..., :, y_start:y_end]
    else:  # If it wraps around, we concatenate the two slices
        cols = torch.cat([rows[..., :, y_start:], rows[..., :, :y_end]], dim=-1)

    return cols


def circular_add_2d(tensor, pos_i, pos_j, size_i, size_j, adder=None):
    H, W = tensor.shape[-2:]
    x_start = pos_i % H
    x_end = (pos_i + size_i) % H

    if x_end >= x_start:
        row_ind = torch.arange(x_start, x_end)  # No wrap-around in rows
    else:
        row_ind = torch.cat((torch.arange(x_start, H), torch.arange(0, x_end)))  # Wrap-around in rows

    y_start = pos_j % W
    y_end = (pos_j + size_j) % W

    if y_end >= y_start:
        col_ind = torch.arange(y_start, y_end)  # No wrap-around in columns
    else:
        col_ind = torch.cat((torch.arange(y_start, W), torch.arange(0, y_end)))  # Wrap-around in columns

    if adder is not None:
        tensor[..., row_ind[:, None], col_ind] += adder
        # tensor[row_ind][:, col_ind] += adder  Makes a copy!!!
    return tensor[..., row_ind[:, None], col_ind]


@torch.no_grad()
def multi_diffusion(net, noise, class_labels, randn_like=torch.randn_like, num_steps=18, sigma_min=0.002, sigma_max=80,
                    rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, guidance=4.0,
                    frame_h=64, frame_w=64, overlap=32, randomize=15, save_trajectory=False):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    if guidance:
        class_labels = torch.cat([class_labels, torch.zeros_like(class_labels)], dim=0)

    # Main sampling loop.
    result = noise.to(torch.float64) * t_steps[0]
    total_h, total_w = noise.shape[-2:]

    predictions = torch.zeros_like(result, requires_grad=False)
    counts = torch.zeros(1, 1, *result.shape[-2:], device=result.device)
    cy = (total_h - frame_h - 1) // (frame_h - overlap) + 1
    cx = (total_w - frame_w - 1) // (frame_w - overlap) + 1
    y, x = (total_h - frame_h) // max(1, cy), (total_w - frame_w) // max(1, cx)
    mod_y, mod_x = (total_h - frame_h) % max(1, cy), (total_w - frame_w) % max(1, cx)

    assert randomize < overlap // 2  # true overlap is frame_w - (x+1)
    trajectory = []
    for it, (t_cur, t_next) in enumerate(list(zip(t_steps[:-1], t_steps[1:]))):  # 0, ..., N-1
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        predictions.fill_(0)
        counts.fill_(0)

        for ci in range(cy + 1):
            for cj in range(cx + 1):
                i = y * ci + min(ci, mod_y)
                j = x * cj + min(cj, mod_x)
                if randomize > 0:
                    if ci != 0 and ci != cy:
                        i += int(np.random.rand() * (2 * randomize + 1)) - randomize
                    elif ci == 0:
                        i -= int(np.random.rand() * (randomize + 1))
                    elif ci == cy:
                        i += int(np.random.rand() * (randomize + 1))
                    if cj != 0 and cj != cx:
                        j += int(np.random.rand() * (2 * randomize + 1)) - randomize
                    elif cj == 0:
                        j -= int(np.random.rand() * (randomize + 1))
                    elif cj == cx:
                        j += int(np.random.rand() * (randomize + 1))

                x_next = get_wrapped_window(result, i, j, frame_h, frame_w)
                crt_labels = get_wrapped_window(class_labels, i, j, frame_h, frame_w)
                x_next = run_one_time_step(net, x_next, crt_labels, t_cur, t_next, gamma, S_noise, randn_like, guidance,
                                           second_order=(it < num_steps - 1))

                circular_add_2d(predictions, i, j, frame_h, frame_w, adder=x_next)
                circular_add_2d(counts, i, j, frame_h, frame_w, adder=1)
        assert counts.min() > 0

        new_result = predictions / counts
        if save_trajectory:
            d_cur = (new_result - result) / (t_next - t_cur)
            trajectory.append(d_cur.cpu())
        result = new_result

    if save_trajectory:
        return result, torch.cat(trajectory, dim=0)
    else:
        return result


@torch.no_grad()
def edm_sampler_guided(
        net, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, guidance=3.0, only_euler=False):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    if guidance:
        class_labels = torch.cat([class_labels, torch.zeros_like(class_labels)], dim=0)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        if guidance:
            x_h2 = torch.cat([x_hat, x_hat], dim=0)
            t_h2 = t_hat
            denoised = net(x_h2, t_h2, class_labels).to(torch.float64)
            d_cur = (x_h2 - denoised) / t_h2
            d_cond, d_uncond = torch.chunk(d_cur, 2, dim=0)
            d_cur = torch.lerp(d_uncond, d_cond, weight=guidance)
        else:
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat

        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1 and not only_euler:
            if guidance:
                x_n2 = torch.cat([x_next, x_next], dim=0)
                denoised = net(x_n2, t_next, class_labels).to(torch.float64)
                d_prime = (x_n2 - denoised) / t_next
                d_cond, d_uncond = torch.chunk(d_prime, 2, dim=0)
                d_prime = torch.lerp(d_uncond, d_cond, weight=guidance)
            else:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


@torch.no_grad()
def noise_mixing_edm_edit(
        net, original_noise, original_dirs, class_labels=None,
        num_steps=250, sigma_min=0.002, sigma_max=80, rho=7,
        guidance=3.0, mixing_alpha=0.3, only_euler=True, enforce_om=True):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    assert only_euler is True  # 2nd order method work poorly on edits

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=original_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = original_noise.to(torch.float64) * t_steps[0]
    if guidance:
        class_labels = torch.cat([class_labels, torch.zeros_like(class_labels)], dim=0)

    guide_x = x_next
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Don't add noise in edit
        t_hat = net.round_sigma(t_cur)
        x_hat = x_cur
        guide_d = original_dirs[-1 - i]  # Traverse from the end to beginning
        guide_x = guide_x + (t_next - t_hat) * guide_d

        # Euler step.
        if guidance:
            x_h2 = torch.cat([x_hat, x_hat], dim=0)
            t_h2 = t_hat  # torch.cat([t_hat, t_hat], dim=0)
            denoised = net(x_h2, t_h2, class_labels).to(torch.float64)
            d_cur = (x_h2 - denoised) / t_h2
            d_cond, d_uncond = torch.chunk(d_cur, 2, dim=0)
            d_cur = torch.lerp(d_uncond, d_cond, weight=guidance)
        else:
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
        d_guided = torch.lerp(d_cur, guide_d, weight=((num_steps - i) / num_steps) ** mixing_alpha)

        d_cur = d_guided
        x_next = x_hat + (t_next - t_hat) * d_cur

        if type(enforce_om) is torch.Tensor:
            x_next = enforce_om * guide_x + (1 - enforce_om) * x_next
        elif enforce_om:
            x_next = class_labels[:1, :1] * guide_x + (1 - class_labels[:1, :1]) * x_next

    return x_next


def ddim_invert_fpi(
        net, x0, class_labels=None,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        guidance=3.0, fpi_iter=5):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=x0.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
    t_steps = t_steps.flip(dims=[-1])

    # Main sampling loop.
    if guidance:
        class_labels = torch.cat([class_labels, torch.zeros_like(class_labels)], dim=0)

    x_next = x0
    directions = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # (0, 0.002), (0.002, 0.004), ...
        x_cur = x_next

        # Do NOT increase noise temporarily: deterministic version
        t_hat = t_cur
        x_hat = x_cur

        # Fixed Point Iterations
        # x_hat is the fixed point; x_cur is z_{t-1}
        d_cur = 0  # Initialize to save in the end
        for _ in range(fpi_iter):
            # Euler step.
            if guidance:
                x_h2 = torch.cat([x_hat, x_hat], dim=0)
                t_h2 = t_next  # torch.cat([t_hat, t_hat], dim=0)
                denoised = net(x_h2, t_h2, class_labels).to(torch.float64)
                d_cur = (x_h2 - denoised) / t_h2
                d_cond, d_uncond = torch.chunk(d_cur, 2, dim=0)
                d_cur = torch.lerp(d_uncond, d_cond, weight=guidance)
            else:
                denoised = net(x_hat, t_next, class_labels).to(torch.float64)
                d_cur = (x_hat - denoised) / t_next
            x_hat = x_cur + (t_next - t_hat) * d_cur
        directions.append(d_cur)
        x_next = x_hat
    return x_next, t_steps[-1], directions


def run_one_time_step(net, x_cur, class_labels, t_cur, t_next, gamma, S_noise, randn_like, guidance, second_order=False,
                      **extra_args):
    # Increase noise temporarily.
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

    # Euler step.
    if guidance:
        x_h2 = torch.cat([x_hat, x_hat], dim=0)
        t_h2 = t_hat  # Scalar
        denoised = net(x_h2, t_h2, class_labels, **extra_args).to(torch.float64)
        d_cur = (x_h2 - denoised) / t_h2
        d_cond, d_uncond = torch.chunk(d_cur, 2, dim=0)
        d_cur = torch.lerp(d_uncond, d_cond, weight=guidance)
    else:
        denoised = net(x_hat, t_hat, class_labels, **extra_args).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat

    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if second_order:
        if guidance:
            x_n2 = torch.cat([x_next, x_next], dim=0)
            denoised = net(x_n2, t_next, class_labels, **extra_args).to(torch.float64)
            d_prime = (x_n2 - denoised) / t_next
            d_cond, d_uncond = torch.chunk(d_prime, 2, dim=0)
            d_prime = torch.lerp(d_uncond, d_cond, weight=guidance)
        else:
            denoised = net(x_next, t_next, class_labels, **extra_args).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def create_lanczos_filter_square(size, cutoff_freq, a=3):
    # Create a grid of (x, y) coordinates
    center = size // 2
    R = np.arange(size) - center
    R_norm = R * cutoff_freq

    # Apply the Lanczos formula
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle divide by zero gracefully
        lanczos_filter = np.where(R_norm < a, np.sinc(R_norm) * np.sinc(R_norm / a), 0)

    # 2D
    lanczos_filter = np.outer(lanczos_filter, lanczos_filter)
    # Normalize the filter to preserve energy
    lanczos_filter /= np.sum(lanczos_filter)

    return lanczos_filter


def low_pass_lanczos(tensor, cutoff=0.1, order=2, filter_size=21):
    lanczos_filter = create_lanczos_filter_square(filter_size, cutoff, order)
    pt_lanczos = torch.from_numpy(lanczos_filter)[None, None].to(tensor.device, torch.float32)
    channels = tensor.shape[-3]
    pt_lanczos = pt_lanczos.expand(channels, -1, -1, -1)
    out = F.pad(tensor, (filter_size // 2,) * 4, mode='circular')
    out = F.conv2d(out, pt_lanczos, groups=channels)
    return out


def noise_uniformization_technique(big, ih, iw, cutoff=0.1, down=(32, 32), kernel=63, shuffle=True):
    assert big.shape[0] == 1
    c = big.shape[1]
    px = big.shape[-2] // ih
    py = big.shape[-1] // iw
    split_noise = big.reshape(c, px, ih, py, iw).permute(1, 3, 0, 2, 4).reshape(px * py, c, ih, iw)

    # orig_lf = blur(split_noise, kernel_size=kernel, sigma=sigma, padding='circular', rescale=True)
    orig_lf = low_pass_lanczos(split_noise, cutoff=cutoff, order=32, filter_size=kernel)
    style = torch.nn.functional.interpolate(orig_lf[:1], down, mode='nearest').expand(px * py, -1, -1, -1)

    style = style.flatten(start_dim=-2)  # b x c x N
    if shuffle:
        perms = torch.rand(px * py, down[0] * down[1], device=big.device).argsort(dim=1)
    else:
        perms = torch.arange(down[0] * down[1], device=big.device).expand(px * py, -1)
    perms[0] = torch.arange(down[0] * down[1])  # Fix the first to the prototype
    style = torch.gather(style, dim=-1, index=perms[:, None].expand_as(style)).reshape(px * py, -1, *down)
    style = torch.nn.functional.interpolate(style, (ih, iw), mode='nearest')
    latents = (split_noise - orig_lf) + style
    latents = latents.reshape(px, py, c, ih, iw).permute(2, 0, 3, 1, 4).reshape(1, c, px * ih, py * iw)
    return latents


class PostProcessor:
    def __init__(self):
        self.decoder = None
        self.scale = np.float32(0.5) / np.float32([4.17, 4.62, 3.71, 3.28])
        self.bias = np.float32(0) - np.float32([5.81, 3.25, 0.12, -2.15]) * self.scale

    def load_decoder(self, images):
        if self.decoder is None and images.shape[1] == 4:
            self.decoder = SDLatentVAE(device=images.device)
        return self

    def load_sd(self, device):
        self.decoder = SDLatentVAE(device=device)
        return self

    def tiled_decode(self, tensor, base_size=128, desired_off=16, sf=8, verbose=False):
        # For very large images, we recommend also synchronizing Group Norm statistics, as in ScaleCrafter.
        # see https://github.com/YingqingHe/ScaleCrafter
        num_win = (np.array(tensor.shape[-2:]) + base_size - 1) // base_size
        base_size = np.array(tensor.shape[-2:]) // num_win

        off = desired_off // num_win * num_win
        stride = np.array(base_size, dtype=np.int64) - off // num_win
        fold_kern = stride + off
        final_size = np.array(tensor.shape[-2:]) * sf
        wins = torch.nn.functional.unfold(tensor, fold_kern, stride=stride)[0].T.view(-1, 4, *fold_kern).type(
            torch.float32)
        if verbose:
            print("Actual offset", off, ", wins.shape", wins.shape)

        # Linear blending
        off_sf = off * sf
        row_indices = torch.arange(fold_kern[0].item() * sf, device=tensor.device, dtype=torch.int32)
        col_indices = torch.arange(fold_kern[1].item() * sf, device=tensor.device, dtype=torch.int32)

        # Generate the 2D grid (row-column order)
        grid_i, grid_j = torch.meshgrid(row_indices, col_indices, indexing='ij')
        gi_rew = torch.maximum(torch.relu(off_sf[0] - grid_i), torch.relu(grid_i - fold_kern[0] * sf + off_sf[0] + 1))
        gj_rew = torch.maximum(torch.relu(off_sf[1] - grid_j), torch.relu(grid_j - fold_kern[1] * sf + off_sf[1] + 1))
        linear_w = 1 - torch.maximum(gi_rew / (off_sf[0] + 1), gj_rew / (off_sf[1] + 1))

        decoded_chunks = [self.decoder.decode(win[None]) for win in wins]
        decoded = torch.cat(decoded_chunks, dim=0)
        decoded = torch.cat([decoded * linear_w, torch.ones_like(decoded[:, :1]) * linear_w], dim=1)
        folded = torch.nn.functional.fold(decoded.flatten(start_dim=1).T[None], tuple(final_size), fold_kern * 8,
                                          stride=stride * 8)
        folded = folded[:, :3] / folded[:, 3:]
        return folded

    def decode(self, latents, max_size=256, desired_off=16):
        if latents.shape[1] == 10:  # SVBRDF
            return latents[:, :3]  # Return diffuse part
        if latents.shape[1] != 4:
            return latents
        self.load_decoder(latents)
        with torch.no_grad():
            if latents.shape[-1] * latents.shape[-2] > max_size * max_size:
                decoded = self.tiled_decode(latents, max_size, desired_off)
            else:
                decoded = self.decoder.decode(latents.type(torch.float32))
        return decoded

    def to_numpy(self, images, max_size=128):
        decoded = self.decode(images, max_size=max_size)
        return (decoded * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    def encode(self, image):
        x = self.decoder.encode(image)
        return x
