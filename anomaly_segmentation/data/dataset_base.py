import abc
from pathlib import Path
from typing import Tuple, List
import numpy as np

import torch
from PIL import ImageOps, Image
from torch.utils.data import Dataset

from anomaly_segmentation.op_utils import get_resize_transform


class DatasetBase(Dataset):
    def __init__(self, name, object_name, data_root, resize=512, n_clusters=5):
        super(DatasetBase, self).__init__()
        self.name = name
        self.object_name = object_name
        self.data_root = Path(data_root)
        self.resize = resize
        self.n_clusters = n_clusters
        self.img_paths, self.inv_paths, self.gt_paths = self.load_dataset_folder()
        self.transform_x = get_resize_transform(resize, normalize=True)
        self.transform_mask = get_resize_transform(resize, normalize=False)

    def __len__(self):
        return len(self.img_paths)

    def load_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.exif_transpose(img)
        return self.transform_x(img)

    def load_gt_mask(self, path):
        if path is None:
            return torch.zeros([self.resize, self.resize], dtype=torch.uint8)
        img = torch.tensor(np.array(Image.open(path).convert('RGB')), dtype=torch.float32)
        close = torch.abs(img[..., None, :] - self.color_map().float()).mean(dim=-1)
        labels = torch.argmin(close, dim=-1).type(torch.uint8)
        return labels

    def load_inv_mask(self, path):
        # Inv paths are path to binary image masks that tell which part of the images should be excluded
        # Useful if you capture textured objects on various backgrounds which should be ignored
        if path is None:
            return torch.zeros([self.resize, self.resize], dtype=torch.float32)
        mask = Image.open(path).convert('RGB')
        mask = self.transform_mask(mask)
        return mask.mean(dim=0)

    @abc.abstractmethod
    def load_dataset_folder(self) -> Tuple[List, List, List]:
        """ Load paths """

    def path_tokens(self, img_path):
        img_name = str(img_path).split('/')[-1]
        return self.object_name, "u", img_name

    def tokens_at(self, idx):
        return self.path_tokens(self.img_paths[idx])

    @staticmethod
    def color_map():
        return torch.tensor([
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 255],
            [0, 255, 0],
            [0, 0, 255]], dtype=torch.uint8)
