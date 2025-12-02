import random
from pathlib import Path
from typing import Tuple

from PIL import Image
import tifffile as tiff

import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import contingency_matrix
from torch import Tensor
from torchvision import transforms as T
from matplotlib import cm

def score_to_reds(np_im, cmap=True, norm=True):
    if cmap:
        if norm:
            np_im = (np_im - np_im.min()) / (np_im.max() - np_im.min())
        cma = cm.Reds(np_im, bytes=True)
        image = Image.fromarray(cma, 'RGBA')
    else:
        image = Image.fromarray(np_im)
    return image.convert('RGB')

def get_resize_transform(resize, normalize=False):
    transforms = [T.Resize((resize, resize), T.InterpolationMode.BILINEAR), T.ToTensor()]
    if normalize:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)

def scale_features(tensor: torch.tensor) -> torch.tensor:
    # tensor shape B x C x H x W
    tf = tensor.flatten(start_dim=-2)
    mini = tf.min(dim=-1).values[..., None, None]
    maxi = tf.max(dim=-1).values[..., None, None]
    div = (maxi - mini + 1e-8)
    return (tensor - mini) / div


def blur(image, kernel_size=7, sigma=None):
    if sigma is None:
        sigma = kernel_size / 4
    shape = image.shape
    im_b = image[(None,) * (4 - len(shape))]
    return torchvision.transforms.functional.gaussian_blur(im_b, kernel_size, sigma=sigma).view(shape)


def get_normal_values(image: torch.tensor, blur_size: int) -> torch.tensor:
    # As in Aittala 2016 and Zhao 2020 (guessed diffuse map)
    x_mean = blur(image, kernel_size=blur_size)
    delta = image - x_mean
    x_sigma = torch.sqrt(blur(torch.square(delta), kernel_size=blur_size))
    x_star = delta / (x_sigma + 1e-4)
    return x_star


def _normalize_in_range(lab_t: torch.tensor, blur_size: int):
    tf = lab_t.flatten(start_dim=2)  # B x C x H*W
    mini = tf.min(dim=-1).values[..., None, None]  # B x C x 1 x 1
    maxi = tf.max(dim=-1).values[..., None, None]  # B x C x 1 x 1
    out = scale_features(get_normal_values(lab_t, blur_size))  # scaled in [0, 1]
    out = out * (maxi - mini) + mini
    return out


def load_image_tensor(path, image_size, color_space="LAB", normalize="no", blur_size=201, device=None):
    input_image = cv2.imread(str(path))
    assert normalize in ["none", "brightness", "all"]
    assert color_space in ["LAB", "RGB", "YCrCb", "GRAY"]
    hw = input_image.shape[:2]
    if image_size == "original":
        image_size = hw
    else:
        image_size = tuple(image_size)
    if normalize == "none":
        if color_space == "LAB":
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
        elif color_space == "RGB":
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        elif color_space == "YCrCb":
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
        elif color_space == "GRAY":
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)[..., None]
        return F.interpolate(image_to_tensor(input_image, device=device), size=image_size, mode='bilinear'), hw
    else:
        lab_t = image_to_tensor(cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB), device=device)
        lab_t = F.interpolate(lab_t, size=image_size, mode='bilinear')
        if normalize == "brightness":
            lab_t[:, :1] = _normalize_in_range(lab_t[:, :1], blur_size)
        else:
            lab_t = _normalize_in_range(lab_t, blur_size)
        if color_space == "LAB":
            return lab_t, hw
        elif color_space == "RGB":
            color_i = cv2.cvtColor(tensor_to_image(lab_t), cv2.COLOR_LAB2RGB)
        elif color_space == "GRAY":
            return lab_t[:, :1], hw
        else:
            color_i = cv2.cvtColor(cv2.cvtColor(tensor_to_image(lab_t), cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2YCrCb)
        return image_to_tensor(color_i, device=device), hw


def image_to_tensor(image, device):
    image = torch.tensor(image, device=device, dtype=torch.float32)
    image = image.permute(2, 0, 1)[None] / 255
    return image


def tensor_to_image(image: torch.tensor):
    out = image.detach().cpu()
    out = (out.squeeze(0).permute(1, 2, 0) * 255)
    return out.numpy().astype(np.uint8)


def build_edge_mask(image, tile_size):
    H, W = image.shape[-2:]
    input_ones = image.new_ones((1, 1, H + tile_size[0] // 2 * 2, W + tile_size[1] // 2 * 2))
    divisor = F.fold(F.unfold(input_ones, kernel_size=tile_size), output_size=input_ones.shape[-2:],
                     kernel_size=tile_size)
    divisor = divisor[0, 0, tile_size[0] // 2:-(tile_size[0] // 2), tile_size[1] // 2:-(tile_size[1] // 2)]
    return divisor / divisor.max()


def reflect_pad(image, patch_size=7):
    p = patch_size // 2
    return torch.nn.functional.pad(image, (p, p, p, p), mode='reflect')


class NormalizeFeatures(torch.nn.Module):
    def __call__(self, x):
        return F.normalize(x, dim=1)  # Normalize channel dimension


class ScaleUniformFeatures(torch.nn.Module):
    def __call__(self, x):
        return scale_features(x)


class RelativeToMeanFeatures(torch.nn.Module):
    def __call__(self, x):
        return x - torch.mean(x, dim=(-2, -1), keepdim=True)


def log_alpha(alpha, path_tokens, scale=True, path="logs/alpha"):
    obj_name, class_name, img_name = path_tokens
    out_path = Path(path) / obj_name / class_name
    out_path.mkdir(parents=True, exist_ok=True)
    if scale:
        alpha = scale_features(alpha[None, None])[0, 0]
    else:
        assert 0 <= alpha.min().item() <= alpha.max().item() <= 1
    gray = (alpha.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(gray).save(out_path / img_name.replace('png', 'jpg'))


def clear_borders(alphas, border=5):
    if border <= 0:
        return alphas
    alphas[..., :border, :] = 0
    alphas[..., -border:, :] = 0
    alphas[..., :, :border] = 0
    alphas[..., :, -border:] = 0
    return alphas


def save_clusters(dataset, labels):
    path_tokens = [dataset.path_tokens(path) for path in dataset.img_paths]
    obj_names, class_names, img_names = zip(*path_tokens)
    df = pd.DataFrame({'object': obj_names,
                       'class': class_names,
                       'img_name:': img_names,
                       'label': labels})
    df.to_csv('img_assignment.csv', mode='a')


def unravel_indices(
        indices: Tensor,
        shape: Tuple[int, ...],
) -> Tensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


@torch.no_grad()
def dist_weight(delta, sigma):
    return torch.exp(-0.5 * (delta / sigma) ** 2)


def compute_cluster_f1(gt_labels, pred_labels, average='micro'):
    # Compute the contingency matrix
    contingency = contingency_matrix(gt_labels, pred_labels)

    # Use Hungarian matching to find the best cluster assignments
    row_ind, col_ind = linear_sum_assignment(-contingency)

    contingency[:, row_ind] = contingency[:, col_ind]
    print(contingency)

    # Reassign the predicted labels based on the best cluster assignments
    pred_labels_mapped = np.zeros_like(pred_labels)
    for i, j in zip(row_ind, col_ind):
        pred_labels_mapped[pred_labels == j] = i

    return f1_score(gt_labels, pred_labels_mapped, average=average)


def save_alpha_unit(alphas, dataset, normalize=True, vis_size=None):
    print("Saving alphas normalized per unit")
    root_dir = Path("logs/alpha_unit")
    for idx, alpha in enumerate(alphas):
        obj_name, class_name, img_name = dataset.path_tokens(dataset.img_paths[idx])
        out_path = root_dir / obj_name
        out_path.mkdir(parents=True, exist_ok=True)
        if normalize:
            alpha = scale_features(alpha[None, None])[0, 0]
        gray = (alpha.cpu().numpy() * 255).astype(np.uint8)
        gray = Image.fromarray(gray)
        if vis_size is not None:
            gray = gray.resize(vis_size)
        gray.save(out_path / f"{class_name}_{img_name}".replace('png', 'jpg'))


def save_alpha_tiff(alphas, dataset, dest_path="logs/alpha_tiff"):
    print("Saving alphas in tiff format for evaluation")
    root_dir = Path(dest_path)
    if type(alphas) is torch.Tensor:
        alphas = alphas.cpu().numpy()
    if type(alphas[0]) is torch.Tensor:
        alphas = [alpha.cpu().numpy() for alpha in alphas]
    for idx, alpha in enumerate(alphas):
        obj_name, class_name, img_name = dataset.path_tokens(dataset.img_paths[idx])
        img_name = img_name.split('.')[0]
        out_path = root_dir / obj_name
        out_path.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(out_path / f"{class_name}_{img_name}.tiff", alpha)


def log_output_image(dataset, idx, image, path, vis_size, resample=1, ext='jpg'):
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    obj_name, class_name, img_name = dataset.tokens_at(idx)
    img_stem = img_name.split('.')[0]
    save_path = save_dir / f'{class_name}_{img_stem}.{ext}'
    Image.fromarray(image).resize(tuple(vis_size), resample=resample).save(save_path)


def loss_hard(q=0.9):
    def loss_hard_q(pred, out):
        mse = torch.nn.functional.mse_loss(pred, out, reduction='none').view(-1)
        q99 = torch.quantile(mse, q=q, dim=-1)
        mse = torch.masked_fill(mse, mse < q99, 0)
        loss = mse.sum() / (mse.numel() * (1 - q))
        return loss

    return loss_hard_q


class MulticlassDiceLoss(torch.nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """

    def __init__(self, num_classes, softmax_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.softmax_dim = softmax_dim

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """
        probabilities = logits
        if self.softmax_dim is not None:
            probabilities = torch.nn.functional.softmax(logits, dim=self.softmax_dim)

        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)

        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # prredicted probability for the actual class.
        intersection = (targets_one_hot * probabilities).sum()

        mod_a = intersection.sum()
        mod_b = targets.numel()

        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.log()
        return dice_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
