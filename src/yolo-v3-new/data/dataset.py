from typing import Any

import numpy as np
import os
import pandas as pd

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

Target = list[dict[str, Any]]

class LocalDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(
            np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
        ).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        target = [] 

        for bbox in bboxes:
            target.append({"box": bbox[0:4], "class": bbox[4]})

        return image, target
