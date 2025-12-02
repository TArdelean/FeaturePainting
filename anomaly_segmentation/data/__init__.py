import torch.utils.data

from .mvtec import MVTecDataset
from .simple_dataset import SimpleDataset
import torch
from PIL import Image
import tifffile as tiff

from torchvision import transforms as T
from pathlib import Path

class IndicesDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        super(IndicesDataset, self).__init__()
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, item):
        return self.tensor[item]


class CustomDataLoader:
    def __init__(self, feature_provider, dataset, *args, device='cuda:0', batch_size=4, **kwargs):
        self.dataset = dataset
        self.feature_provider = feature_provider.init(self.dataset)
        self.data_loader = torch.utils.data.DataLoader(dataset, *args, batch_size=batch_size, **kwargs)
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        self.iter = iter(self.data_loader)
        return self

    def __next__(self):
        data = next(self.iter)
        idx = data[-1]
        features = self.feature_provider.get(idx)
        return features, *data

    def __len__(self):
        return self.data_loader.__len__()

    def feature_loader(self, shuffle=True, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        indices = IndicesDataset(torch.arange(0, len(self.dataset), device=self.device))
        data_loader = torch.utils.data.DataLoader(indices, batch_size=batch_size, shuffle=shuffle)
        for chunk in data_loader:
            features = self.feature_provider.get(chunk)
            yield features

class MaskMemory:
    def __init__(self, masks_root, dataset, device, is_tiff=True):
        self.masks_root = Path(masks_root)
        self.ss_transform = T.ToTensor()
        self.is_tiff = is_tiff
        self.memory = self.load(dataset).to(device)

    def load(self, dataset: MVTecDataset):
        memory = []
        mask_dir = self.masks_root / dataset.object_name
        for img_path in dataset.img_paths:
            obj_name, class_name, img_name = dataset.path_tokens(img_path)
            if self.is_tiff:
                self_path = mask_dir / f"{class_name}_{img_name.split('.')[0]}.tiff"
                ss_target = tiff.imread(self_path)
            else:
                self_path = mask_dir / f"{class_name}_{img_name.split('.')[0]}.jpg"
                ss_target = Image.open(self_path)
            ss_target = (self.ss_transform(ss_target)).float()
            memory.append(ss_target)
        return torch.stack(memory)

    def get(self, indices: torch.tensor):
        return self.memory[indices]