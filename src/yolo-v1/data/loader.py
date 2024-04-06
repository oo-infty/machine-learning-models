from typing import Optional, Callable, Any

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection

from data.transform import DataTransform, ImagePredictionTransform, ImageTrainingTransform, ParseTarget, ToTargetTensor

class CustomVOCDetection(Dataset):
    def __init__(
        self,
        root: str,
        image_set: str,
        image_transforms: Optional[Callable] = None,
        data_transforms: Optional[Callable] = None,
        *,
        enable_target_transform: bool = True,
        size: int,
        bounding_box: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.inner_dataset = VOCDetection(
            root,
            image_set=image_set,
            target_transform=ParseTarget()
        )

        self.image_transforms= image_transforms
        self.data_transforms = data_transforms

        if enable_target_transform:
            self.target_transform = ToTargetTensor(448, size, bounding_box, classes)
        else:
            self.target_transform = None

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.inner_dataset[index]
        new_img, new_target = img, target

        if self.image_transforms is not None:
            new_img, new_target = self.image_transforms([new_img, new_target])

        if self.data_transforms is not None:
            new_img, new_target = self.data_transforms([new_img, new_target])

        if self.target_transform is not None:
            new_target = self.target_transform(new_target)

        return new_img, new_target

    def get_item(self, index: int) -> tuple[Any, Any, Any, Any]:
        img, target = self.inner_dataset[index]
        new_img, new_target = img, target

        if self.image_transforms is not None:
            new_img, new_target = self.image_transforms([new_img, new_target])
            img, target = self.image_transforms([img, target])

        if self.data_transforms is not None:
            new_img, new_target = self.data_transforms([new_img, new_target])

        if self.target_transform is not None:
            new_target = self.target_transform(new_target)

        return new_img, new_target, img, target

def voc_detection_dataset(
    image_set: str,
    image_transforms: Optional[Callable] = None,
    data_transforms: Optional[Callable] = None,
    *,
    enable_target_transform: bool = True,
    size: int,
    bounding_box: int,
    classes: int,
) -> CustomVOCDetection:
    return CustomVOCDetection(
        "data",
        image_set=image_set,
        image_transforms=image_transforms,
        data_transforms=data_transforms,
        size=size,
        bounding_box=bounding_box,
        classes=classes,
        enable_target_transform=enable_target_transform,
    )

def loader(
    image_set: str,
    batch_size: int,
    size: int,
    bounding_box: int,
    classes: int,
    enable_target_transform: bool = True,
) -> DataLoader:
    image_transform = ImageTrainingTransform() if image_set == "train" else ImagePredictionTransform()
    dataset = voc_detection_dataset(
        image_set,
        image_transform,
        DataTransform(),
        size=size,
        bounding_box=bounding_box,
        classes=classes,
        enable_target_transform=enable_target_transform,
    )
    return DataLoader(dataset, batch_size, True)
