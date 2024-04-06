from typing import Any

import torch
import torchvision
from torch import Tensor
from torchvision.transforms.v2 import Transform, Compose
from torchvision.transforms.v2 import RandomHorizontalFlip, Resize
from torchvision.transforms.v2 import Normalize, ToDtype, ToImage
from torchvision.tv_tensors import BoundingBoxes

import data.label

class ParseTarget(Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, label: dict[str, Any]) -> dict[str, Any]:
        label = label["annotation"]
        size = (int(label["size"]["height"]), int(label["size"]["width"]))
        objects = label["object"]
        boxes = []
        classes = []
        names = []
    
        for obj in objects:
            box = obj["bndbox"]
            boxes.append([int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])])
            classes.append(data.label.label_to_value(obj["name"]))
            names.append(obj["name"])

        bounding_box = BoundingBoxes(boxes, format="XYXY", canvas_size=size)
        return {"boxes": bounding_box, "classes": classes, "names": names}

class ToTargetTensor(Transform):
    def __init__(
        self,
        img_size: int,
        size: int,
        bounding_box: int,
        classes: int
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.size = size
        self.bounding_box = bounding_box
        self.classes = classes

    def forward(self, input: dict[str, Any]) -> Tensor:
        target = torch.zeros((self.size, self.size, self.classes + 5 * self.bounding_box))
        classes = torch.as_tensor(input["classes"])
        boxes = torch.as_tensor(input["boxes"]).reshape(-1, 4) / self.img_size
        boxes = torchvision.ops.box_convert(boxes, "xyxy", "cxcywh")
        grid_idx = torch.floor(boxes[:, 0:2] * self.size).to(torch.int)

        for i, (gx, gy) in enumerate(grid_idx):
            box = boxes[i]
            new_box = Tensor([
                (box[0] - gx / self.size) * self.size,
                (box[1] - gy / self.size) * self.size,
                box[2],
                box[3],
            ]).repeat(self.bounding_box)
            target[gx, gy, self.classes + self.bounding_box:] = new_box
            target[gx, gy, classes[i]] = 1.0
            target[gx, gy, self.classes:self.classes + self.bounding_box] = 1.0

        return target


class ImageTrainingTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

        self.transforms = Compose([
            Resize((448, 448)),
            RandomHorizontalFlip(),
            ToImage(),
        ])

    def forward(self, *inputs: Any) -> Any:
        return self.transforms(*inputs)

class ImagePredictionTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

        self.transforms = Compose([
            Resize((448, 448)),
            ToImage(),
        ])

    def forward(self, *inputs: Any) -> Any:
        return self.transforms(*inputs)

class DataTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

        self.transforms = Compose([
            ToDtype(torch.float, scale=True),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def forward(self, *inputs: Any) -> Any:
        return self.transforms(*inputs)

