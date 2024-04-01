import torch
from torch.utils.data import Dataset
import transforms
import numpy as np
import pandas as pd


class ICML(Dataset):
    def __init__(self, spatial_transform: transforms.Compose) -> None:
        super().__init__()
        self.spatial_transform = spatial_transform

        # columns: "emotion", " Usage", " pixels"
        df = pd.read_csv("icml_face_data.csv")
        images = [
            np.array(list(map(int, row.split(" "))), np.uint8).reshape(48, 48)
            for row in df[" pixels"]
        ]
        images = np.stack(images)
        images = np.repeat(images[:, :, :, np.newaxis], repeats=3, axis=-1)
        self.data = images  # shape: (35887, 48, 48, 3)
        self.label: pd.Series[int] = df["emotion"]
        assert len(self.data) == len(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imgs = [self.spatial_transform(x) for x in self.data]
        imgs = torch.stack(imgs, 0)

        fake_audio = torch.randn((10, 156))
        return fake_audio, imgs
