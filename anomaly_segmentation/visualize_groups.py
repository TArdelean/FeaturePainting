from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import op_utils
from data.cl_pairs_ds import PairsDataset
from data.dataset_base import DatasetBase
from data.feature_provider import FeatureProvider


def labels_to_maps(pairs_ds):
    label_maps = []
    rgb_maps = []
    l_idx = 0
    for ind in range(len(pairs_ds.groups)):
        label_map = torch.zeros(pairs_ds.hw, dtype=torch.uint8, device="cpu")
        for j, group in enumerate(pairs_ds.groups[ind]):
            label_map[group[:, 0], group[:, 1]] = j
            l_idx += 1
        label_maps.append(label_map)

        cmap = plt.get_cmap('tab20', label_map.max() + 1)
        rgb = (cmap(label_map)[..., :3] * 255).astype(np.uint8)
        rgb[label_map == 0] = 0

        rgb_maps.append(rgb)

    return rgb_maps


@hydra.main(version_base=None, config_path="conf", config_name="train_cl")
def my_app(cfg: DictConfig) -> None:
    cfg.track = "visualize_groups"

    dataset: DatasetBase = instantiate(cfg.dataset)
    fp: FeatureProvider = instantiate(cfg.features).init(dataset)

    pairs_path = Path(cfg.pairs_dataset_path)
    pairs_ds = PairsDataset(fp, dataset, pairs_path)

    rgb_maps = labels_to_maps(pairs_ds)

    for idx in range(len(dataset)):
        colors = rgb_maps[idx]
        op_utils.log_output_image(dataset, idx, colors, 'vis', (512, 512), resample=0, ext='jpg')


if __name__ == "__main__":
    my_app()
