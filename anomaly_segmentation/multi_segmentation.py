import json

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment
from sklearn import cluster
from tqdm import tqdm

import op_utils
from data import CustomDataLoader


# Inspired from https://github.com/mhamilton723/STEGO/blob/master/src/modules.py#L134
class MinibatchKMeans(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(MinibatchKMeans, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def inner_products(self, x):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
        return inner_products

    def forward(self, x, alpha, log_probs=False):
        inner_products = self.inner_products(x)  # B x N x H x W
        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs


def fit_minibatch_kmeans(data_loader, f_dim, ep=10):
    cluster_probe = MinibatchKMeans(f_dim, data_loader.dataset.n_clusters).to(data_loader.device)
    optimizer = torch.optim.Adam(cluster_probe.parameters(), lr=0.001)
    feature_loader = data_loader.feature_loader(batch_size=4, shuffle=True)
    for _ in range(ep):
        for features in feature_loader:
            loss, cluster_probs = cluster_probe.forward(features, alpha=None, log_probs=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    labels = []
    with torch.no_grad():
        ordered_loader = data_loader.feature_loader(batch_size=4, shuffle=False)
        for features in ordered_loader:
            label = torch.argmax(cluster_probe.inner_products(features), dim=1)  # B x H x W
            labels.append(label.flatten(1, 2))
    return torch.cat(labels, dim=0)  # ALL x H*W


def fit_linear_probe(data_loader, f_dim, hw, ep=7):
    data_loader.data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=4, shuffle=True)
    linear_probe = nn.Conv2d(f_dim, data_loader.dataset.n_clusters, kernel_size=1).to(data_loader.device)
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=0.001)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = op_utils.MulticlassDiceLoss(num_classes=data_loader.dataset.n_clusters, softmax_dim=1)
    for _ in range(ep):
        for features, _, _, gt_mask, idx in data_loader:
            gt_mask = F.interpolate(gt_mask[:, None].to(data_loader.device), size=hw, mode='nearest')  # B x 1 x H x W
            loss = loss_fn(linear_probe(features), gt_mask[:, 0].type(torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    labels = []
    with torch.no_grad():
        ordered_loader = data_loader.feature_loader(batch_size=4, shuffle=False)
        for features in ordered_loader:
            label = torch.argmax(linear_probe(features), dim=1)  # B x H x W
            labels.append(label.flatten(1, 2))
    return torch.cat(labels, dim=0)  # ALL x H*W


def assigned_metrics(contingency):
    assert contingency.shape[0] == contingency.shape[1]  # Not yet implemented
    # confusion_matrix: tensor[pred_clusters_size, gt_clusters_size]
    tp = torch.diag(contingency)
    fp = torch.sum(contingency, dim=0) - tp
    fn = torch.sum(contingency, dim=1) - tp

    acc = torch.sum(tp) / torch.sum(contingency)
    prc = tp / (tp + fp)
    iou = (tp / (tp + fp + fn)).mean()
    f1 = (2 * tp / (2 * tp + fn + fp))
    # f1_micro = (2 * tp.sum() / (2 * tp.sum() + fn.sum() + fp.sum())) == acc
    return {'acc': acc.item(), 'iou': iou.item(), 'f1_macro': f1.mean().item(), 'f1': f1.tolist()}


def remap_labels(labels: torch.Tensor, assignments) -> torch.Tensor:
    # assignments: Tuple[range(len(gt_clusters)), pred_clusters]
    vals = labels.new_tensor(assignments[1]).argsort()
    return vals[labels.type(torch.long)]

def get_contingency(dataset, cluster_labels, hw):
    gt_cs, pred_cs = dataset.n_clusters, dataset.n_clusters
    contingency = torch.zeros(gt_cs * pred_cs, device=cluster_labels.device, dtype=torch.long)
    for _, _, gt_mask, idx in tqdm(dataset):
        gt_mask = F.interpolate(gt_mask[None, None].to(cluster_labels.device), size=hw, mode='nearest').view(-1)
        pairs = gt_mask.type(torch.long) * pred_cs + cluster_labels[idx]
        contingency += torch.bincount(pairs.flatten(), minlength=len(contingency))
    contingency = contingency.reshape(gt_cs, pred_cs)
    assignments = linear_sum_assignment(contingency.cpu(), maximize=True)
    contingency[:, assignments[0]] = contingency[:, assignments[1]]
    return contingency, assignments

def compute_metrics(data_loader):
    metrics = {'probe': {}, 'cluster': {}}
    C, h, w = data_loader.feature_provider.get(0).shape[-3:]
    feature_stream = []
    for features in data_loader.feature_loader(batch_size=4, shuffle=False):
        feature_stream.append(features)
    feature_stream = torch.cat(feature_stream, dim=0)  # ALL x C x H x W
    feature_stream = feature_stream.permute(0, 2, 3, 1).reshape(-1, C)  # ALL*H*W x C
    cm = cluster.KMeans(n_clusters=data_loader.dataset.n_clusters, random_state=42, n_init=10)
    cm.fit(feature_stream.cpu().numpy())

    cluster_labels = torch.tensor(cm.labels_, device=feature_stream.device, dtype=torch.uint8)
    cluster_labels = cluster_labels.reshape(len(data_loader.dataset), -1)  # ALL x H*W

    contingency, assignments = get_contingency(data_loader.dataset, cluster_labels, (h, w))
    metrics['cluster'] = assigned_metrics(contingency)

    # Kmeans batches
    cluster_labels_kmb = fit_minibatch_kmeans(data_loader, f_dim=C)
    contingency_kmb, assignments_kmb = get_contingency(data_loader.dataset, cluster_labels_kmb, (h, w))
    metrics['cluster_kmb'] = assigned_metrics(contingency_kmb)

    # Linear Probe
    cluster_labels_lin = fit_linear_probe(data_loader, f_dim=C, hw=(h, w))
    contingency_lin, assignments_lin = get_contingency(data_loader.dataset, cluster_labels_lin, (h, w))
    metrics['cluster_lin'] = assigned_metrics(contingency_lin)

    return metrics, remap_labels(cluster_labels, assignments).reshape(-1, h, w)


def multi_segmentation(dataset, fp, should_compute_metrics, vis_size, n_clusters=None, save_tiff=False):
    op_utils.set_seed(42)
    device = fp.device
    data_loader = CustomDataLoader(fp, dataset, batch_size=1)
    if not should_compute_metrics:
        assert n_clusters is not None
        feature_stream = []
        print(f"{'-' * 10}\nCollect Features\n{'-' * 10}")
        for features, img, inv_mask, gt_mask, idx in tqdm(data_loader):
            with torch.no_grad():
                inv_mask = torch.nn.functional.interpolate((inv_mask[:, None]).to(device).float(), features.shape[-2:])
                features = features.permute(0, 2, 3, 1)
                features = features[(inv_mask[:, 0] < 0.5)]

            feature_stream.append(features)
        print(f"{'-' * 10}\nCompute KMeans\n{'-' * 10}")
        feature_stream = torch.cat(feature_stream, dim=0)
        cm = cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        all_labels = cm.fit_predict(feature_stream.cpu().numpy())

        print(f"{'-' * 10}\nSave segmentation maps\n{'-' * 10}")
        color_map = dataset.color_map().numpy()
        for _, _, _, idx in tqdm(dataset):
            features = fp.get([idx])
            fr = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            labels = cm.predict(fr.cpu().numpy())

            colors = color_map[labels].reshape(*features.shape[-2:], 3).astype(np.uint8)
            op_utils.log_output_image(dataset, idx, colors, 'vis', vis_size, resample=0, ext='png')
        if save_tiff:
            op_utils.save_alpha_tiff(all_labels.reshape(-1, *features.shape[-2:]), dataset, "tiff_labels")

    if should_compute_metrics:
        print(f"{'-' * 10}\nCompute metrics\n{'-' * 10}")
        metrics, cluster_labels = compute_metrics(data_loader)
        with open('metrics.json', "w") as f:
            json.dump(metrics, f, indent=4)
        color_map = dataset.color_map().to(device)
        for idx in range(len(dataset)):
            colors = color_map[cluster_labels[idx]].cpu().numpy()
            op_utils.log_output_image(dataset, idx, colors, 'vis', vis_size, resample=0, ext='png')
        op_utils.save_alpha_tiff(cluster_labels.cpu(), dataset, "tiff_labels")
    print(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

# noinspection DuplicatedCode
@hydra.main(version_base=None, config_path="conf", config_name="segmentation")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dataset)
    fp = instantiate(cfg.features)

    multi_segmentation(dataset, fp, cfg.compute_metrics, cfg.vis_size, cfg.n_clusters, cfg.save_labels_tiff)


if __name__ == "__main__":
    my_app()
