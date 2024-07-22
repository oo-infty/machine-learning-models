from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import config
from data.dataset import LocalDataset, Target
from data.transform import train_transforms, test_transforms

def get_loader_impl(
    csv_path: str,
    img_dir: str,
    label_dir: str,
    transform: Callable,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    def collate_fn(data: list[tuple[Tensor, Target]]):
        return torch.stack([t[0] for t in data]), [t[1] for t in data]

    dataset =  LocalDataset(
        csv_file=csv_path,
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

    return loader

def get_loader(csv_path: str, split: str):
    if split == "train":
        transform = train_transforms(config.IMAGE_SIZE)
        shuffle = True
    else:
        transform = test_transforms(config.IMAGE_SIZE)
        shuffle = False

    return get_loader_impl(
        csv_path=csv_path,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        transform=transform,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=shuffle,
    )

def get_loaders(train_csv_path, test_csv_path):
    train_loader = get_loader(train_csv_path, "train")
    test_loader = get_loader(test_csv_path, "test")
    validate_loader = get_loader(train_csv_path, "validate")
    return train_loader, test_loader, validate_loader
