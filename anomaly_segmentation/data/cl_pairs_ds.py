from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import v2

from torch.utils.data import Dataset


class PairsDataset(Dataset):
    def __init__(self, feature_provider, dataset, cl_path, n1=20, n2=20, k=3):
        """
        Args:
            feature_provider:
            dataset: Original dataset (contains the images)
            cl_path: Path to the contrastive learning dataset metadata
            n1: Number of points from the source image
            n2: Number of points from the friends / antagonists / complementary negatives
            k: Number of friend / antagonist groups sampled
        """
        super(PairsDataset, self).__init__()

        self.dataset = dataset
        self.feature_provider = feature_provider.init(self.dataset)
        self.hw = self.feature_provider.get(0).shape[-2:]
        self.cl_path = Path(cl_path)
        self.n1 = n1
        self.n2 = n2
        self.k = k
        self.groups = np.load(str(self.cl_path / 'groups.npy'), allow_pickle=True)
        self.friends = np.load(str(self.cl_path / 'friends.npy'), allow_pickle=True)
        self.antagonists = np.load(str(self.cl_path / 'antagonists.npy'), allow_pickle=True)

        self.color_jitter = v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        self.blur = v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))

    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def augmented_image(self, im_idx, pts, f_hw=(64, 64)):
        img = Image.open(self.dataset.img_paths[im_idx]).convert('RGB')
        aug_pts = pts.clone()
        if np.random.rand() < 0.5:  # Vertical Flip
            img = v2.functional.vertical_flip(img)
            aug_pts[:, 0] = f_hw[0] - aug_pts[:, 0] - 1
        if np.random.rand() < 0.5:  # Horizontal Flip
            img = v2.functional.horizontal_flip(img)
            aug_pts[:, 1] = f_hw[1] - aug_pts[:, 1] - 1
        img = self.color_jitter(img)
        img = self.dataset.transform_x(img)
        if np.random.rand() < 0.5:
            img = self.blur(img)
        return img, aug_pts

    def __getitem__(self, im_idx):
        src_img = im_idx
        group_idx = np.random.randint(0, len(self.groups[im_idx]))
        pts_idx = np.random.choice(len(self.groups[im_idx][group_idx]), self.n1, replace=True)
        src_pts = torch.tensor(self.groups[im_idx][group_idx][pts_idx])
        aug_img, aug_pts = self.augmented_image(im_idx, src_pts, self.hw)

        friends_idx = np.random.choice(len(self.friends[im_idx][group_idx]), self.k, replace=False)
        friends = self.friends[im_idx][group_idx][friends_idx]  # nd.array[(im_idx, group_idx])

        pos_img = friends[:, 0]
        pos_pts = []
        for pos_im_idx, pos_group_idx in friends:
            group = self.groups[pos_im_idx][pos_group_idx]
            fpi = np.random.choice(len(group), self.n2, replace=True)
            pos_pts.append(group[fpi])

        pos_pts = torch.tensor(np.asarray(pos_pts))
        ant_img, ant_pts = self.random_antagonists(self.antagonists[im_idx][group_idx], self.k, self.n2)

        return src_img, src_pts, pos_img, pos_pts, ant_img, ant_pts, aug_img, aug_pts

    def random_antagonists(self, group_pairs, k, n):
        pairs_idx = np.random.choice(len(group_pairs), k, replace=False)
        sel_pairs = group_pairs[pairs_idx]  # nd.array[(im_idx, group_idx])
        sel_img = sel_pairs[:, 0]
        pts = []
        for img_idx, group_idx in sel_pairs:
            group = self.groups[img_idx][group_idx]
            point_idx = np.random.choice(len(group), n, replace=True)
            pts.append(group[point_idx])
        return sel_img, torch.tensor(np.asarray(pts))
