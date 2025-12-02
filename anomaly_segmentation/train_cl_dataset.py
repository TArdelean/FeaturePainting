from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import op_utils
from data.cl_pairs_ds import PairsDataset
from data.feature_provider import FeaturesFromFolder, FeatureProvider
from multi_segmentation import multi_segmentation
from anomaly_segmentation.models import ContrastNet


def compute_contrastive_loss(src_sel, pos_sel, ant_sel, tau=0.4):
    # src_sel: B x 1 x N1 x D
    # pos_sel/ant_sel: B x K x N2 x D
    def n1(x):
        return torch.nn.functional.normalize(x, dim=-1)

    src_sel = src_sel.flatten(1, 2)  # B x N1 x D
    src_mean = torch.mean(src_sel, dim=1, keepdim=True)  # B x 1 x D
    pos_mean = torch.mean(pos_sel, dim=2, keepdim=False)  # B x K x D
    src_plus_pos = torch.cat([src_mean, pos_mean], dim=1)  # B x (1+K) x D
    plus_mean = torch.cat([src_mean, torch.mean(src_plus_pos, dim=1, keepdim=True)], dim=1)  # B x (1+1) x D
    vvt = torch.einsum('bsd,bpd->bsp', n1(src_sel), n1(plus_mean)) / tau  # B x N1 X (1+K)

    minus_mean = torch.mean(ant_sel, dim=2, keepdim=False)  # B x K x D

    vvm = torch.einsum('bsd,bpd->bsp', n1(src_sel), n1(minus_mean)) / tau  # B x N1 x K
    sum_vvm = torch.sum(torch.exp(vvm), dim=-1)  # B x N1

    combs = torch.exp(vvt) + sum_vvm[:, :, None]  # B x N1 x (1+K)  i.e. sum of enemy exps + positive exp
    xent = vvt - torch.log(combs)  # B x N1 x (1+K)

    wts = torch.ones_like(xent, requires_grad=False)
    xent = xent * wts

    loss = -xent.mean()
    return loss


def sample_tensor(tensor, points):
    """
    Get the features from a tensor at specific 2d points
    :param tensor: B1 x B2 x C x H x W
    :param points: B1 x B2 x N x 2
    :return: B1 x B2 x N x C
    """
    *b, c, h, w = tensor.shape
    tensor = tensor.reshape(-1, c, h * w)  # B x C x HW
    points = points[..., 0] * w + points[..., 1]  # B1 x B2 x N
    points = points.reshape(-1, 1, points.shape[-1]).expand(-1, c, -1)  # B x C x N
    gathered = torch.gather(tensor, dim=-1, index=points).permute(0, 2, 1)  # B x N x C
    return gathered.reshape(*b, -1, c)  # ... x N x C


def train_contrast_iteration(net, src_f, pts, pos_f, pos_pts, ant_f, ant_pts, aug_f, aug_pts):
    """
    Make an iteration of contrastive learning
    :param net: nn.Module
    :param src_f: B x C x H x W
    :param pts: B x N1 x 2
    :param pos_f: B x K x C x H x W
    :param pos_pts: B x K x N2 x 2
    :param ant_f: B x K x C x H x W
    :param ant_pts: B x K x N2 x 2
    :param aug_f: B x C x H x W
    :param aug_pts: B x N1 x 2
    :return: The loss
    """
    k = pos_f.shape[1]
    f = torch.cat([src_f[:, None], pos_f, ant_f, aug_f[:, None]], dim=1)
    projected = net(f.flatten(0, 1), fun='proj')  # B(1+K+K+1) x D x H x W
    projected = projected.reshape(src_f.shape[0], -1, *projected.shape[-3:])  # B x (1+K+K+1) x D x H x W
    src_proj, pos_proj, ant_proj, aug_proj = torch.split(projected, [1, k, ant_f.shape[1], 1], dim=1)
    src_sel = sample_tensor(src_proj, pts.to(src_f.device))  # B x 1 x N1 x D
    pos_sel = sample_tensor(pos_proj, pos_pts.to(src_f.device))  # B x K x N2 x D
    ant_sel = sample_tensor(ant_proj, ant_pts.to(src_f.device))  # B x K x N2 x D
    aug_sel = 0  # Disable augmentations

    loss = compute_contrastive_loss(src_sel, pos_sel, ant_sel)
    return loss


def train_cl(pairs_ds: PairsDataset, device, tag, cache_dir='cache', iterations=2000, epochs=None):
    print("Computing contrastive features for", pairs_ds.dataset.name)

    net = ContrastNet(512).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    cache_path = Path(cache_dir) / pairs_ds.dataset.name / tag
    data_loader = DataLoader(pairs_ds, batch_size=4, shuffle=True, num_workers=8)

    if epochs is None:
        epochs = (iterations - 1) // len(pairs_ds) + 1

    losses = []
    for _ in tqdm(range(epochs)):
        for src_idx, src_pts, pos_idx, pos_pts, ant_idx, ant_pts, aug_img, aug_pts in data_loader:
            src_f = pairs_ds.feature_provider.get(src_idx)
            pos_f = pairs_ds.feature_provider.get(pos_idx)
            ant_f = pairs_ds.feature_provider.get(ant_idx)
            aug_f = pairs_ds.feature_provider.fe(aug_img.to(device))
            loss = train_contrast_iteration(net, src_f, src_pts, pos_f, pos_pts, ant_f, ant_pts, aug_f, aug_pts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    plt.plot(losses)
    plt.savefig("loss.pdf", format="pdf")

    net = net.eval()
    for i in range(len(pairs_ds.dataset)):
        with torch.no_grad():
            features = pairs_ds.feature_provider.get([i])
            projected = net(features, fun='emb')

        obj_name, class_name, img_name = pairs_ds.dataset.tokens_at(i)
        (cache_path / obj_name).mkdir(parents=True, exist_ok=True)
        file_path = cache_path / obj_name / f"{class_name}_{img_name.split('.')[0]}.pt"
        torch.save(projected[0], file_path)
    torch.save(net.state_dict(), cache_path / f"{pairs_ds.dataset.object_name}.pt")


@hydra.main(version_base=None, config_path="conf", config_name="train_cl")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    op_utils.set_seed(42)

    device = torch.device(cfg.device)
    dataset = instantiate(cfg.dataset)
    fp: FeatureProvider = instantiate(cfg.features)

    cl_pairs_path = Path(cfg.pairs_dataset_path)
    pairs_ds = PairsDataset(fp, dataset, cl_pairs_path, n1=1000, n2=100, k=cfg.cl_k)

    train_cl(pairs_ds, device, tag=cfg.tag, cache_dir=cfg.cache_dir, iterations=cfg.iterations)
    if cfg.segment_after:
        my_fp = FeaturesFromFolder(cfg.tag, cache_dir=cfg.cache_dir, save_in_memory=fp.save_in_memory, device=fp.device)
        multi_segmentation(dataset, my_fp, should_compute_metrics=cfg.compute_metrics,
                           vis_size=cfg.vis_size, n_clusters=cfg.n_clusters, save_tiff=True)


if __name__ == "__main__":
    my_app()
