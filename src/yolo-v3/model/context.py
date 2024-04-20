from typing import NamedTuple

import torch
from torch import Tensor

from model.network import YoloNetwork, YoloNetworkResult


class SplitTensorResult(NamedTuple):
    """Result of YoloContext.split_tensor

    Args:
        boxes (Tensor): bounding boxes
        confidence (Tensor): confidence of boxes
        classes (Tensor): probabilities of classes of the object in the box
    """

    boxes: Tensor
    confidence: Tensor
    classes: Tensor

class ComposedSplitTensorResult(NamedTuple):
    """Result of YoloContext.split_tensor of all feature maps

    Args:
        small (SplitTensorResult): result of the small feature map
        intermediate (SplitTensorResult): result of the intermediate feature map
        large (SplitTensorResult): result of the large feature map
    """

    small: SplitTensorResult
    intermediate: SplitTensorResult
    large: SplitTensorResult

class YoloContext:
    """An object which stores all essential constants and the network structure, along
    with some relevant helper functions

    Args:
        device (str): where to run the model ("gpu", e.g.)
        num_box (int): the number of boxes of each grid
        num_class (int): the number of classes
        anchor_boxes (Tensor): boxes on which predictions are based
    """

    def __init__(
        self,
        device: str,
        num_box: int,
        num_class: int,
        anchor_boxes: Tensor,
    ) -> None:
        self.device = device

        self.num_box = num_box
        self.num_class = num_class
        self.anchor_boxes = anchor_boxes

        self.network = YoloNetwork(num_box, num_class).to(device)

    def split_tensor(self, tensor: Tensor) -> SplitTensorResult:
        """Split the network's output to tensors of bounding boxes, confidence and
        probabilities.

        Args:
            tensor (Tensor): the network's output

        Returns:
            SplitTensorResult: a tuple consists of relevant tensors
        """

        return SplitTensorResult(
            boxes=tensor[:, :, :, :, 0:4],
            confidence=tensor[:, :, :, :, 4],
            classes=tensor[:, :, :, :, 5:],
        )

    def preprocess_output(
        self,
        output: YoloNetworkResult,
    ) -> ComposedSplitTensorResult:
        """Split the network's output for all feature maps
        
        Args:
            output (YoloNetworkResult): the network's output

        Returns:
            ComposedSplitTensorResult: preprocessed output
        """

        return ComposedSplitTensorResult(
            small=self.split_tensor(output.small),
            intermediate=self.split_tensor(output.intermediate),
            large=self.split_tensor(output.large),
        )

    def decode_bounding_box(
            self,
            boxes: Tensor,
            offset: Tensor,
            anchor: Tensor,
            size: Tensor,
        ) -> Tensor:
        """Transform the tensor of boxes in raw format to adjusted anchor boxes

        Args:
            boxes (Tensor): boxes in a raw output form
            offset (Tensor): grid cells' top left corners' offset from the origin
            anchor (Tensor): anchor boxes
            size (Tensor): size of the corresponding feature map

        Returns:
            Tensor: bounding boxes of the "CXCYWH" style, relative to the origin
        """

        return torch.stack(
            [
                (torch.sigmoid(boxes[:, :, :, :, 0]) + offset) / size,
                (torch.sigmoid(boxes[:, :, :, :, 1]) + offset) / size,
                torch.exp(boxes[:, :, :, :, 2]) * anchor[:, 0],
                torch.exp(boxes[:, :, :, :, 3]) * anchor[:, 1],
            ]
        )

    def sigmoid_inverse(self, x: Tensor) -> Tensor:
        """Inverse of the sigmoid function
        
        Args:
            x (Tensor): the input tensor

        Returns:
            Tensor: the output tensor
        """

        return torch.log(x / (1 - x))

    def encode_bounding_box(
        self,
        boxes: Tensor,
        offset: Tensor,
        anchor: Tensor,
        size: Tensor,
    ) -> Tensor:
        """Convert the CXCYWH-styled boxes to raw format
            
        Args:
            boxes (Tensor): CXCYWH-styled boxes, which is relative to the origin
            offset (Tensor): grid cells' top left corners' offset from the origin
            anchor (Tensor): anchor boxes
            size (Tensor): size of the corresponding feature map

        Returns:
            Tensor: bounding boxes in raw format
        """

        return torch.stack(
            [
                self.sigmoid_inverse(boxes[:, :, :, :, 0] * size - offset),
                self.sigmoid_inverse(boxes[:, :, :, :, 1] * size - offset),
                torch.log(boxes[:, :, :, :, 2] / anchor[:, 0]),
                torch.log(boxes[:, :, :, :, 3] / anchor[:, 1]),
            ]
        )
