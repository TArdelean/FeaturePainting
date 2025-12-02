import torch
import torch.nn.functional as F

from fca import StatFCA, generate_all_sets, get_gaussian_w


def reference_median_masked(features, val_mask, patch_size, chunk_size=64):
    # Compute the full median (overlapping patches) by median of medians (to preserve memory)
    generator = generate_all_sets(features, patch_size, chunk_size)
    medians = []
    for f_set, m_chunk in zip(generator, torch.split(val_mask, chunk_size, dim=0)):
        fvalues, _ = torch.sort(f_set, dim=-1)  # h x W x C x T**2
        fvalues = fvalues.flatten(0, 1)[m_chunk.flatten() > 0.5]  # n x C x T**2
        if fvalues.shape[0] == 0:
            continue
        medians.append(torch.median(fvalues, dim=0).values)  # C x T**2
    return torch.median(torch.stack(medians, dim=0), dim=0).values  # C x T**2


class FCAWithMask(StatFCA):
    def __init__(self, patch_size, sigma_p=None, k_s=5, sigma_s=1.0, chunk_size=8,
                 reference_selector=reference_median_masked):
        super(FCAWithMask, self).__init__(patch_size, sigma_p, k_s, sigma_s, chunk_size, reference_selector)

    def __call__(self, features, val_mask):
        r_set = self.reference_selector(features, val_mask, self.patch_size)
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
