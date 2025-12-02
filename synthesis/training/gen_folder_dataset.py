import re
from pathlib import Path

import numpy as np
import tifffile as tiff
import torch.utils.data
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2

from .dataset import Dataset


class GenFolderDataset(Dataset):
    def __init__(self, path='datasets/folder', resolution=512,
                 object_name='tile', split='test',
                 labels_path="datasets/GT_labels/mvtec",
                 post_encoding_channels=4,
                 **super_kwargs):
        super(Dataset, self).__init__()
        resolution = (resolution, resolution)
        self.data_root = Path(path)
        self.object_name = object_name
        self.labels_path = None if labels_path is None else Path(labels_path)
        self.image_paths = self.load_paths()
        self.label_maps = self.load_label_maps()
        self.num_cat = self.compute_num_cat()
        self.transform = v2.Compose([
            v2.RandomResizedCrop(size=resolution, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.25, contrast=(0.8, 1.4), saturation=0.25, hue=0.1),
            # v2.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.label_resolution = [64, 64]
        raw_shape = [len(self.image_paths), post_encoding_channels, *resolution]
        super().__init__(name=f"gen_folder_{self.data_root.name}", raw_shape=raw_shape, **super_kwargs)
        # self._raw_labels = np.array(self.labels, dtype=np.int64)
        self._raw_labels = np.zeros(len(self.image_paths), dtype=np.int64)
        self.original_resolution = self._load_raw_image(0).shape[-2:]

    @staticmethod
    def _file_ext(fname):
        return Path(fname).suffix

    @property
    def label_shape(self):
        return [self.num_cat]

    def close(self):
        self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        image = read_image(str(self.image_paths[raw_idx]))

        return image  # CHW

    def _load_raw_labels(self):
        return self._raw_labels

    def compute_num_cat(self):
        return max(np.max(arr) for arr in self.label_maps) + 1

    def load_paths(self):
        img_paths = self.load_dataset_folder()
        return img_paths

    def load_dataset_folder(self):
        images_dir = self.data_root
        paths = sorted(images_dir.iterdir(), key=lambda x: int(re.findall(r'\d+', x.stem)[-1]))
        return paths

    def load_label_maps(self):
        if self.labels_path is None:
            return None
        label_maps = []
        for img_path in self.image_paths:
            img_name = img_path.name

            lm_path = self.labels_path / self.object_name / f"u_{img_name.split('.')[0]}.tiff"
            label_map = tiff.imread(str(lm_path))
            label_maps.append(label_map)
        return label_maps

    def get_label(self, idx, resolution=None):
        if self.labels_path is None:
            return super(GenFolderDataset, self).get_label(idx)
        raw_idx = self._raw_idx[idx]
        raw_map = self.label_maps[raw_idx]
        label_map = torch.tensor(raw_map, dtype=torch.int64)

        label_map = torch.nn.functional.one_hot(label_map, num_classes=self.num_cat)  # H x W x C
        label_map = label_map.permute(2, 0, 1)  # C x H x W
        if resolution is None:
            resolution = self.label_resolution
        # desired_resolution = self.original_resolution if original_resolution else self.label_resolution
        if label_map.shape[-2] != resolution:
            return v2.functional.resize_mask(label_map, resolution)
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
        label = tv_tensors.Mask(self.get_label(idx, resolution=image.shape[-2:]))
        image = tv_tensors.Image(image)
        image, label = self.transform(image, label)
        label = v2.functional.resize_mask(label, self.label_resolution)
        return image, label
