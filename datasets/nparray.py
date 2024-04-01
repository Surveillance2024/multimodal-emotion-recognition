from typing import List
import torch
from torch.utils.data import Dataset
import transforms
import numpy as np


class NumpyArray(Dataset):
    def __init__(
        self,
        images: np.ndarray | List[np.ndarray],
        spatial_transform: transforms.Compose,
    ) -> None:
        super().__init__()
        self.data = images
        self.spatial_transform = spatial_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imgs = [self.spatial_transform(x) for x in self.data]
        imgs = torch.stack(imgs, 0)

        fake_audio = torch.randn((10, 156))
        return fake_audio, imgs
