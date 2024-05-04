from typing import Any

import torch
from torchvision.transforms.v2 import Transform, Compose
from torchvision.transforms.v2 import (
    RandomPhotometricDistort,
    RandomResizedCrop,
    RandomHorizontalFlip,
    SanitizeBoundingBoxes,
    Resize,
)
from torchvision.transforms.v2 import Normalize, ToDtype, ToImage
from torchvision.tv_tensors import BoundingBoxes

import data.label


class ParseTarget(Transform):
    """Parse the target in a dict format from Pascal VOC"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, target: dict[str, Any]) -> dict[str, Any]:
        """The target from Pascal VOC dataset

        Args:
            target (dict[str, Any]): original targets

        Returns:
            dict[str, Any]: parsed target containing necessary fields
        """

        target = target["annotation"]
        size = (int(target["size"]["height"]), int(target["size"]["width"]))
        objects = target["object"]
        boxes = []
        classes = []
        names = []

        for obj in objects:
            box = obj["bndbox"]
            boxes.append(
                [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])]
            )
            classes.append(data.label.label_to_value(obj["name"]))
            names.append(obj["name"])

        bounding_box = BoundingBoxes(boxes, format="XYXY", canvas_size=size)
        return {"boxes": bounding_box, "classes": classes, "names": names}


class ImageTrainingTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

        self.transforms = Compose(
            [
                ToImage(),
                RandomPhotometricDistort(p=1),
                RandomHorizontalFlip(),
                RandomResizedCrop((416, 416)),
                SanitizeBoundingBoxes(labels_getter=lambda d: d[1]["boxes"]),
            ]
        )

    def forward(self, *inputs: Any) -> Any:
        return self.transforms(*inputs)


class ImagePredictionTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

        self.transforms = Compose(
            [
                ToImage(),
                Resize((416, 416)),
            ]
        )

    def forward(self, *inputs: Any) -> Any:
        return self.transforms(*inputs)


class DataTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

        self.transforms = Compose(
            [
                ToDtype(torch.float, scale=True),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, *inputs: Any) -> Any:
        return self.transforms(*inputs)
