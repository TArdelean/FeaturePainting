import re

import skimage
import torch.utils.data
from pathlib import Path
from torchvision.io import read_image
from torchvision.transforms import v2
import numpy as np
import tifffile as tiff
from .dataset import Dataset
from torchvision import tv_tensors


class GenMvtecDataset(Dataset):
    def __init__(self, path='datasets/mvtec_anomaly_detection', category='all', resolution=512,
                 object_name='tile', split='test',
                 labels_path="datasets/GT_labels/mvtec",
                 post_encoding_channels=4,
                 **super_kwargs):
        super(Dataset, self).__init__()
        resolution = (resolution, resolution)
        self.data_root = Path(path)
        self.object_name = object_name
        self.split = split
        self.exclude_combined = False
        self.skeletonize_cracks = False
        self.labels_path = None if labels_path is None else Path(labels_path)
        assert category == "all"

        self.image_paths, self.labels = self.load_paths()
        self.label_maps = self.load_label_maps()
        self.transform = v2.Compose([
            # v2.Resize(size=128, antialias=True),
            v2.RandomResizedCrop(size=resolution, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            # v2.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.label_resolution = [64, 64]
        raw_shape = [len(self.image_paths), post_encoding_channels, *resolution]
        super().__init__(name="gen_mvtec", raw_shape=raw_shape, **super_kwargs)
        self._raw_labels = np.array(self.labels, dtype=np.int64)
        self.original_resolution = self._load_raw_image(0).shape[-2:]

    @staticmethod
    def _file_ext(fname):
        return Path(fname).suffix

    def close(self):
        self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        image = read_image(str(self.image_paths[raw_idx]))

        return image  # CHW

    def _load_raw_labels(self):
        return self._raw_labels

    def num_cat(self):
        return len(self.label_names)

    def load_paths(self):
        img_paths, labels, mask_paths = self.load_dataset_folder()
        self.label_names = ["good"] + sorted(set(labels) - {"good"})  # Make sure good is pos 0
        ids = [self.label_names.index(val) for val in labels]

        return img_paths, ids

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

        return img_paths, labels, mask_paths

    def load_label_maps(self):
        if self.labels_path is None:
            return None
        label_maps = []
        for img_path in self.image_paths:
            obj_name, _, class_name, img_name = str(img_path).split('/')[-4:]

            lm_path = self.labels_path / obj_name / f"{class_name}_{img_name.split('.')[0]}.tiff"
            label_map = tiff.imread(lm_path)
            label_maps.append(label_map)

        return label_maps

    def get_label(self, idx, original_resolution=True):
        if self.labels_path is None:
            return super(GenMvtecDataset, self).get_label(idx)
        raw_idx = self._raw_idx[idx]
        raw_map = self.label_maps[raw_idx]
        label_id = self.labels[raw_idx]
        if self.skeletonize_cracks and self.label_names[label_id] == 'crack':
            # raw_map = skimage.morphology.skeletonize(raw_map)
            crack_mask = (raw_map == label_id)
            raw_map = np.where(crack_mask, skimage.morphology.skeletonize(crack_mask) * label_id, raw_map)
        label_map = torch.tensor(raw_map, dtype=torch.int64)

        label_map = torch.nn.functional.one_hot(label_map, num_classes=self.num_cat())  # H x W x C
        label_map = label_map.permute(2, 0, 1)  # C x H x W
        if original_resolution:
            return v2.functional.resize_mask(label_map, self.original_resolution)
        else:
            return label_map

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        # assert isinstance(image, np.ndarray)
        # assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        image = tv_tensors.Image(image)
        label = tv_tensors.Mask(self.get_label(idx))
        image, label = self.transform(image, label)
        label = v2.functional.resize_mask(label, self.label_resolution)
        return image, label
