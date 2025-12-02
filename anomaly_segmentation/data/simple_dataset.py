import re
from typing import Tuple, List

from anomaly_segmentation.data.dataset_base import DatasetBase


class SimpleDataset(DatasetBase):
    def __init__(self, name, data_root, resize=512, object_name='single', mask_suf='_mask', gt_suf='_gt',
                 n_clusters=5, **kwargs):
        self.mask_suf = mask_suf
        self.gt_suf = gt_suf
        super(SimpleDataset, self).__init__(name, object_name, data_root, resize, n_clusters)

    def load_dataset_folder(self) -> Tuple[List, List, List]:

        paths = sorted(list(self.data_root.iterdir()), key=lambda x: int(re.findall(r'\d+', x.stem)[-1]))
        img_paths = [p for p in paths if not p.stem.endswith(self.mask_suf)]
        inv_paths = [self.data_root / f"{p.stem}{self.mask_suf}.png" for p in img_paths]
        inv_paths = [p if p.exists() else None for p in inv_paths]

        gt_paths = [None] * len(paths)

        return img_paths, inv_paths, gt_paths

    def __getitem__(self, idx):
        img_path, inv_path, gt_path = str(self.img_paths[idx]), self.inv_paths[idx], self.gt_paths[idx]

        img = self.load_img(img_path)
        gt_mask = self.load_gt_mask(gt_path)
        inv_mask = self.load_inv_mask(inv_path)

        return img, inv_mask, gt_mask, idx
