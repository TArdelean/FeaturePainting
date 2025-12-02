import re

import numpy as np
import torch
from PIL import Image

from anomaly_segmentation.data.dataset_base import DatasetBase


class MVTecDataset(DatasetBase):

    textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
    nt = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    objects = [*textures]

    def __init__(self, object_name='carpet', resize=512, split="test", exclude_combined=False,
                 data_root="datasets/mvtec_anomaly_detection", name="mvtec", n_clusters=6):
        self.split = split
        self.exclude_combined = exclude_combined
        super().__init__(name, object_name, data_root, resize, n_clusters)

        self.label_names = ["good"] + sorted(set(self.labels) - {"good"})  # Make sure good is pos 0
        self.label_ids = self.build_label_ids()

    def load_gt_mask_mvtec(self, path, idx):
        if path is None:
            return torch.ones([self.resize, self.resize], dtype=torch.uint8) * self.label_names.index('good')
        mask = torch.tensor(np.array(Image.open(path).convert('RGB').resize((self.resize, self.resize))))
        mask = (mask.amax(dim=-1) > 127).type(torch.uint8)
        mask = mask * self.label_ids[idx]
        return mask

    def __getitem__(self, idx):
        img_path, mask_path = str(self.img_paths[idx]), self.gt_paths[idx]

        img = self.load_img(img_path)
        mask = self.load_gt_mask_mvtec(mask_path, idx)

        inv_mask = torch.zeros([self.resize, self.resize], dtype=torch.float32)

        return img, inv_mask, mask, idx

    def build_label_ids(self):
        ids = [self.label_names.index(val) for val in self.labels]
        return np.array(ids)

    def load_dataset_folder(self):
        img_paths, labels, mask_paths = [], [], []

        test_dir = self.data_root / self.object_name / self.split
        gt_dir = self.data_root / self.object_name / 'ground_truth'

        for defect_dir in test_dir.iterdir():
            label = defect_dir.name
            if label == "combined" and self.exclude_combined:
                continue
            paths = sorted(defect_dir.iterdir(), key=lambda x: int(re.findall(r'\d+', x.stem)[-1]))

            length = len(paths)
            img_paths.extend(paths)
            labels.extend([label] * length)

            if label == 'good':
                mask_paths.extend([None] * length)
            else:
                gt_paths = sorted((gt_dir / label).iterdir(), key=lambda x: int(re.findall(r'\d+', x.stem)[-1]))
                mask_paths.extend(gt_paths)

        self.labels = labels
        inv_paths = [None for _ in img_paths]
        return img_paths, inv_paths, mask_paths

    def path_tokens(self, img_path):
        obj_name, _, class_name, img_name = str(img_path).split('/')[-4:]
        return obj_name, class_name, img_name

    def tokens_at(self, idx):
        return self.path_tokens(self.img_paths[idx])
