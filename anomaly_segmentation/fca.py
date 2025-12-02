# Reuses code from https://github.com/TArdelean/AnomalyLocalizationFCA

from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

from op_utils import scale_features


def _get_gaussian_kernel2d(tile_size, sigma, dtype, device):
    sigma = sigma[0]
    mask = torch.zeros((1, 1, tile_size[0] + 3, tile_size[1] + 3), dtype=dtype, device=device)
    mask[:, :, tile_size[0] // 2 + 1, tile_size[0] // 2 + 1] = 1
    mask = blur(mask, tile_size[0], sigma=sigma)
    mask = mask[0, 0, 1:tile_size[0] + 1, 1:tile_size[1] + 1]
    return mask


def reflect_pad(image, patch_size=7):
    p = patch_size // 2
    return torch.nn.functional.pad(image, (p, p, p, p), mode='reflect')


def blur(image, kernel_size, sigma) -> torch.Tensor:
    shape = image.shape
    im_b = image[(None,) * (4 - len(shape))]
    return torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)(im_b).view(shape)


gauss_cache = {}


def get_gaussian_w(device, patch_size, sigma):
    key = (patch_size, sigma)
    if key not in gauss_cache:
        gauss_cache[key] = _get_gaussian_kernel2d(patch_size, [sigma, sigma], torch.float32, device=device)
    return gauss_cache[key]


def generate_all_sets(features, patch_size, chunk_size):
    # Returns chunks of unfolded rows to avoid running out of memory
    b, c, h, w = features.shape
    padded = reflect_pad(features, patch_size[0])
    unf = torch.nn.functional.unfold(padded, (patch_size[0], padded.shape[-1]), stride=(1, 1))
    unf = unf[0].T.reshape(h, c, patch_size[0], padded.shape[-1])
    for i in range(0, len(unf), chunk_size):
        chunk = unf[i:i + chunk_size]
        unf_2 = torch.nn.functional.unfold(chunk, patch_size, stride=(1, 1))
        unf_2 = unf_2.transpose(1, 2).reshape(chunk.shape[0], w, c, -1)
        yield unf_2


def reference_median_tiled(features, patch_size):
    # Median from non-overlapping patches (fast good approximation)
    unf = F.unfold(features, patch_size, stride=patch_size)[0].T.reshape(-1, features.shape[1], *patch_size)
    val, _ = torch.sort(unf.view(-1, features.shape[1], patch_size[0] * patch_size[1]), dim=-1)
    return torch.median(val, dim=0).values


def reference_median_full(features, patch_size, chunk_size=8):
    # Compute the full median (overlapping patches) by median of medians (to preserve memory)
    generator = generate_all_sets(features, patch_size, chunk_size)
    medians = []
    for f_set in generator:
        fvalues, _ = torch.sort(f_set, dim=-1)  # h x W x C x T**2
        medians.append(torch.median(fvalues.flatten(0, 1), dim=0).values)  # C x T**2
    return torch.median(torch.stack(medians, dim=0), dim=0).values  # C x T**2


def reference_mean(features, patch_size):
    return [torch.mean(features.flatten(start_dim=-2), dim=-1)]


def aggregate(fn, arguments, method='min'):
    assert method in ['mean', 'min']
    if method == 'mean':
        acc = arguments[0].new_tensor([0.0])
    else:
        acc = arguments[0].new_tensor([1e8])

    for arg in arguments:
        d_r = fn(arg)
        if method == 'mean':
            acc = d_r + acc
        else:
            acc = torch.minimum(d_r, acc)
    if method == 'mean':
        acc /= len(arguments)
    return acc


class StatMoments:
    def __init__(self, patch_size, powers=(1, 2, 3, 4), sigma=12.5, reference_selector=reference_mean):
        self.patch_size = patch_size
        self.powers = powers
        self.sigma = sigma
        self.reference_selector = reference_selector

    @staticmethod
    def dist(one, many):
        vec_arr = one[:, None, None].expand_as(many)
        loss = torch.mean((vec_arr - many) ** 2, dim=0)
        del vec_arr
        return loss

    def __call__(self, features, agg_method='min'):
        ps = features.new_tensor(list(self.powers))
        b, c, h, w = features.shape
        # features = features - features.mean(dim=1)  # Extract mean to compute central moments
        moments = torch.pow(features[:, :, None, :, :], ps[None, None, :, None, None]).reshape(b, c * len(ps), h, w)
        local_moments = blur(moments, kernel_size=self.patch_size[0], sigma=self.sigma)[0]  # C*powers x H x W
        r_set = self.reference_selector(local_moments, self.patch_size)
        return aggregate(partial(self.dist, many=local_moments), r_set, agg_method)


class StatFCA:
    def __init__(self, patch_size, sigma_p=None, k_s=5, sigma_s=1.0, chunk_size=8,
                 reference_selector=reference_median_tiled):
        self.patch_size = tuple(patch_size)
        self.sigma_p = sigma_p
        self.k_s = k_s
        self.sigma_s = sigma_s
        self.chunk_size = chunk_size
        self.reference_selector = reference_selector
        self.gaussian_mask = None

        assert patch_size[0] == patch_size[1]  # Only implemented for square patches
        self.p_size = (patch_size[0] // 2)
        if self.sigma_s is not None:
            self.local_blur = torchvision.transforms.GaussianBlur(k_s, sigma=self.sigma_s)
        else:
            self.local_blur = None

    def __call__(self, features):
        r_set = self.reference_selector(features, self.patch_size)
        wp = features.shape[-1] + 2 * self.p_size
        generator = generate_all_sets(features, self.patch_size, self.chunk_size)
        if self.sigma_p is not None and self.gaussian_mask is None:
            self.gaussian_mask = get_gaussian_w(features.device, self.patch_size, sigma=self.sigma_p).reshape(-1)
        parts = []
        for f_set in generator:
            fvalues, ind = torch.sort(f_set, dim=-1)  # h x W x C x T**2
            vec_arr = r_set[None, None].expand_as(fvalues)
            loss = F.l1_loss(fvalues, vec_arr, reduction='none')
            loss_re = torch.gather(loss, dim=-1, index=torch.argsort(ind)).mean(dim=2, keepdim=True)  # h x W x 1 x T**2
            if self.sigma_s is not None:
                loss_re = self.local_blur(loss_re.view(-1, 1, *self.patch_size)).reshape(loss_re.shape)
            if self.sigma_p is not None:
                loss_re = loss_re * self.gaussian_mask  # h x W x 1 x T**2
            loss_re = loss_re.permute(0, 2, 3, 1).reshape(f_set.shape[0], -1, features.shape[-1])  # h x 1*T**2 x W
            c_fold = F.fold(loss_re, (self.patch_size[0], wp), kernel_size=self.patch_size)  # h x C x T x WP
            parts.append(c_fold)
        combined = torch.cat(parts, dim=0)  # H x 1 x T x WP
        folded = F.fold(combined.permute(1, 2, 3, 0).reshape(1, -1, features.shape[-2]),
                        output_size=(wp, wp), kernel_size=(self.patch_size[0], wp))
        folded = folded[0, 0, self.p_size:-self.p_size, self.p_size:-self.p_size]  # Remove extra pad -> 1 x 1 x H x W

        if self.sigma_p is not None:
            return folded
        else:
            return folded / (self.patch_size[0] * self.patch_size[1])


def main():
    device = torch.device('cuda')

    normalization_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(-1, 1, 1)
    normalization_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(-1, 1, 1)
    cnn = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval().to(device)
    feature_extractor = torch.nn.Sequential(*list(cnn.children())[:6])

    image = torchvision.io.read_image("000.png").to(device)[None] / 255.0
    with torch.no_grad():
        image = (image - normalization_mean) / normalization_std
        features = feature_extractor(image)
        features = scale_features(features)

    s = StatFCA((9, 9), sigma_p=3.0, sigma_s=1.0)
    a = s(features)
    plt.imshow(a.cpu(), cmap='Reds')
    plt.show()


if __name__ == '__main__':
    main()
