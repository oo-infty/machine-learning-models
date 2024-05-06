from typing import Callable, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection

from data.transform import (
    DataTransform,
    ImagePredictionTransform,
    ImageTrainingTransform,
    ParseTarget,
)


class CustomVOCDetection(Dataset):
    """Custom Pascal VOC dataset with target parsing

    Args:
        root (str): path to store the dataset
        image_set (str): which data split to use
        image_transform (Callable | None): data augmentation transform
        data_transform (Callable | None): normalization transform
    """

    def __init__(
        self,
        root: str,
        image_set: str,
        image_transform: Callable | None = None,
        data_transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.inner_dataset = VOCDetection(
            root,
            image_set=image_set,
            target_transform=ParseTarget(),
        )

        self.image_transforms = image_transform
        self.data_transforms = data_transform

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.inner_dataset[index]

        if self.image_transforms is not None:
            img, target = self.image_transforms([img, target])

        if self.data_transforms is not None:
            img, target = self.data_transforms([img, target])

        return img, target


def loader(
    image_set: str,
    batch_size: int,
) -> DataLoader:
    """Create a Pascal VOC data loader

    Args:
        image_set (str): which data split to use
        batch_size (int): number of samples in one batch

    Returns:
        DataLoader: the data loader
    """

    def collate_fn(data):
        sample = torch.stack([img for (img, _) in data])
        target = [target for (_, target) in data]
        return sample, target

    image_transform = (
        ImageTrainingTransform() if image_set == "train" else ImagePredictionTransform()
    )

    dataset = CustomVOCDetection(
        "data",
        image_set,
        image_transform,
        DataTransform(),
    )

    return DataLoader(
        dataset,
        batch_size,
        True,
        collate_fn=collate_fn,
    )
