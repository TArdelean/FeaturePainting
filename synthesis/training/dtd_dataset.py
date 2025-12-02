import torch.utils.data
from pathlib import Path
from torchvision.io import read_image
from torchvision.transforms import v2
import numpy as np
from .dataset import Dataset


class DTDataset(Dataset):
    def __init__(self, path='datasets/DTD/images', category='all', resolution=512,
                 post_encoding_channels=4,
                 **super_kwargs):
        super(Dataset, self).__init__()
        resolution = (resolution, resolution)
        self.data_root = Path(path)
        if category != 'all':
            self.categories = [category]
        else:
            self.categories = sorted([_.name for _ in self.data_root.iterdir()])
        self.image_paths, self.labels = self.load_paths()
        self.transform = v2.Compose([
            v2.RandomResizedCrop(size=resolution, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.2),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # v2.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            v2.RandomGrayscale(p=0.05),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.label_resolution = [64, 64]

        raw_shape = [len(self.image_paths), post_encoding_channels, *resolution]
        super_kwargs.pop("labels_path", None)
        super_kwargs.pop("object_name")
        super().__init__(name="dtd", raw_shape=raw_shape, **super_kwargs)
        self._raw_labels = np.array(self.labels, dtype=np.int64)

    @staticmethod
    def _file_ext(fname):
        return Path(fname).suffix

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        image = read_image(str(self.image_paths[raw_idx]))
        return image  # CHW

    def _load_raw_labels(self):
        return self._raw_labels

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = np.array(self.labels, dtype=np.int64)
        return self._raw_labels

    def num_cat(self):
        return len(self.categories)

    def load_paths(self):
        paths = []
        labels = []
        for i, category in enumerate(self.categories):
            cat_paths = sorted((self.data_root / category).glob('*.jpg'))
            paths.extend(cat_paths)
            labels.extend([i] * len(cat_paths))
        return paths, labels

    def get_label(self, idx):
        label = self.labels[self._raw_idx[idx]]
        onehot = np.zeros(self.label_shape, dtype=np.float32)
        onehot[label] = 1

        return torch.from_numpy(onehot)[:, None, None].expand(-1, *self.label_resolution)

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._load_raw_image(raw_idx)

        label = self.get_label(idx)
        image = self.transform(image)  # Label is spatially constant -- it doesn't need to be transformed

        assert label.shape[0] == 47
        return image, label
