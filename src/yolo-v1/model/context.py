from enum import Enum

import torch
import torchvision
from torch import Tensor
from torch.nn import Sequential, Module

from model.extractor import YoloInceptionExtractor, YoloExtractor, YoloResNetExtractor
from model.detector import YoloDetector 


class BackboneType(Enum):
    INCEPTION = 0
    RESNET = 1


class YoloContext:
    def __init__(
        self,
        device: str,
        size: int,
        bounding_box: int,
        classes: int,
        confidence_threshold: float,
        iou_threshold: float,
        backbone: BackboneType,
    ) -> None:
        self.device = device
        self.size = size
        self.bounding_box = bounding_box
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.offset_x, self.offset_y = torch.meshgrid(
            torch.arange(self.size),
            torch.arange(self.size),
            indexing="ij",
        )

        self.offset_x = self.offset_x \
            .reshape(self.size, self.size, 1) \
            .repeat_interleave(self.bounding_box, dim=2) \
            .to(device)

        self.offset_y = self.offset_y \
            .reshape(self.size, self.size, 1) \
            .repeat_interleave(self.bounding_box, dim=2) \
            .to(device)

        backbone_net: Module

        match backbone:
            case BackboneType.INCEPTION:
                backbone_net = YoloInceptionExtractor()
            case BackboneType.RESNET:
                backbone_net = YoloResNetExtractor()

        self.network = Sequential(
            backbone_net,
            YoloExtractor(),
            YoloDetector(size, bounding_box, classes)
        ).to(device)

    def split_tensor(self, tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split a tensor of a batch of NN outputs to classes, confidence
        and (x, y, w, h) of each grid.
        """
        
        tensor = tensor.reshape(-1, self.size, self.size, self.classes + 5 * self.bounding_box)
        classes = tensor[:, :, :, :self.classes] 
        confidence = tensor[:, :, :, self.classes:self.classes + self.bounding_box] 
        bounding_box = torch.unflatten(
            tensor[:, :, :, self.classes + self.bounding_box:],
            3,
            (self.bounding_box, 4)
        )

        return classes, confidence, bounding_box

    def convert_bounding_box(self, bounding_box: Tensor) -> Tensor:
        """Convert the bounding boxes' representation to (x1, y1, x2, y2),
        which is relative to the top left corner of the image.
        """

        new_bounding_box = torch.stack([
            (bounding_box[:, :, :, :, 0] + self.offset_x) / self.size,
            (bounding_box[:, :, :, :, 1] + self.offset_y) / self.size, 
            bounding_box[:, :, :, :, 2],
            bounding_box[:, :, :, :, 3]
        ], 4)

        return torchvision.ops \
            .box_convert(new_bounding_box.reshape(-1, 4), "cxcywh", "xyxy") \
            .reshape(-1, self.size, self.size, self.bounding_box, 4)

    def select_class(self, classes: Tensor, confidence: Tensor) -> tuple[Tensor, Tensor]:
        """Select classes for each bounding box with respect to classes
        confidence.
        """

        reshaped_classes = classes.reshape(-1, self.size, self.size, 1, self.classes)
        reshaped_confidences = confidence.reshape(-1, self.size, self.size, self.bounding_box, 1)
        max_class_confidence, indices = (reshaped_confidences * reshaped_classes).max(dim=4)
        return max_class_confidence, indices

    def filter_bounding_box(
        self,
        bounding_box: Tensor,
        class_confidence: Tensor,
        classes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Remove bounding boxes with low confidence and overlapped bounding
        boxes with respect to IoU. Bounding boxes here must be represented as
        (x1, y1, x2, y2).
        """

        num_batches = bounding_box.shape[0]
        batch_id = torch \
            .arange(num_batches) \
            .repeat_interleave(self.size * self.size * self.bounding_box) \
            .to(self.device)

        new_bounding_box = bounding_box.reshape(-1, 4)
        new_class_confidence = class_confidence.reshape(-1)
        new_classes = classes.reshape(-1)

        mask = new_class_confidence > self.confidence_threshold

        batch_id = batch_id[mask]
        new_bounding_box = new_bounding_box[mask]
        new_class_confidence = new_class_confidence[mask]
        new_classes = new_classes[mask]
        
        indices = torchvision.ops.batched_nms(
            new_bounding_box,
            new_class_confidence,
            batch_id,
            self.iou_threshold
        )

        return \
            new_bounding_box[indices], \
            new_classes[indices], \
            new_class_confidence[indices], \
            batch_id[indices]
