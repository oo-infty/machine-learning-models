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
    """

    def __init__(
        self,
        device: str,
        num_box: int,
        num_class: int,
    ) -> None:
        self.device = device

        self.num_box = num_box
        self.num_class = num_class
        self.anchor_boxes: Tensor | None = None

        self.network = YoloNetwork(num_box, num_class).to(device)

        self.size = [
            Tensor([13]).to(device),
            Tensor([26]).to(device),
            Tensor([52]).to(device),
        ]
        self.offset_x = []
        self.offset_y = []

        for size in self.size:
            offset_x, offset_y = self.get_offset(device, int(size.item()), num_box)
            self.offset_x.append(offset_x.to(device))
            self.offset_y.append(offset_y.to(device))

    def set_anchor_boxes(self, anchor_boxes: Tensor) -> None:
        """Set prior anchor boxes

        Args:
            anchor_boxes (Tensor): boxes on which predictions are based
        """

        self.anchor_boxes = anchor_boxes

    def get_offset(
        self,
        device: str,
        size: int,
        num_box: int,
    ) -> tuple[Tensor, Tensor]:
        """Get vertical and horizontal offsets for grids

        Args:
            device (str): where to store the tensors
            size (int): the number of grids in one row
            num_box (int): the number of boxes of each grid

        Returns:
            tuple[Tensor, Tensor]: offset tensors for both directions
        """

        offset_x, offset_y = torch.meshgrid(
            torch.arange(size),
            torch.arange(size),
            indexing="ij",
        )

        offset_x = (
            offset_x.reshape(size, size, 1).repeat_interleave(num_box, dim=2).to(device)
        )

        offset_y = (
            offset_y.reshape(size, size, 1).repeat_interleave(num_box, dim=2).to(device)
        )

        return offset_x, offset_y

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
        *,
        encode_box: bool = False,
        decode_box: bool = False,
    ) -> ComposedSplitTensorResult:
        """Split the network's output for all feature maps

        Args:
            output (YoloNetworkResult): the network's output
            encode_box (bool): whether to encode the box format
            decode_box (bool): whether to decode the box format

        Returns:
            ComposedSplitTensorResult: preprocessed output
        """

        res = []
        res.append(self.split_tensor(output.small))
        res.append(self.split_tensor(output.intermediate))
        res.append(self.split_tensor(output.large))

        if self.anchor_boxes is None:
            raise ValueError("anchor_boxes is None")

        if encode_box and decode_box:
            raise ValueError("encode_box and decode_box cannot be both True")

        if encode_box:
            for i in range(3):
                boxes = self.encode_bounding_box(
                    res[i].boxes,
                    self.offset_x[i],
                    self.offset_y[i],
                    self.anchor_boxes[3 * i : 3 * (i + 1)],
                    self.size[i],
                )

                res[i] = SplitTensorResult(
                    boxes=boxes,
                    confidence=res[i].confidence,
                    classes=res[i].classes,
                )

        if decode_box:
            for i in range(3):
                boxes = self.decode_bounding_box(
                    res[i].boxes,
                    self.offset_x[i],
                    self.offset_y[i],
                    self.anchor_boxes[3 * i : 3 * (i + 1)],
                    self.size[i],
                )

                res[i] = SplitTensorResult(
                    boxes=boxes,
                    confidence=res[i].confidence,
                    classes=res[i].classes,
                )

        return ComposedSplitTensorResult(
            small=res[0],
            intermediate=res[1],
            large=res[2],
        )

    def decode_bounding_box(
        self,
        boxes: Tensor,
        offset_x: Tensor,
        offset_y: Tensor,
        anchor: Tensor,
        size: Tensor,
    ) -> Tensor:
        """Transform the tensor of boxes in raw format to adjusted anchor boxes

        Args:
            boxes (Tensor): boxes in a raw output form
            offset_x (Tensor): grid cells' top left corners' offset from the origin
            offset_y (Tensor): grid cells' top left corners' offset from the origin
            anchor (Tensor): anchor boxes
            size (Tensor): size of the corresponding feature map

        Returns:
            Tensor: bounding boxes of the "CXCYWH" style, relative to the origin
        """

        return torch.stack(
            [
                (torch.sigmoid(boxes[:, :, :, :, 0]) + offset_x) / size,
                (torch.sigmoid(boxes[:, :, :, :, 1]) + offset_y) / size,
                torch.exp(boxes[:, :, :, :, 2]) * anchor[:, 0],
                torch.exp(boxes[:, :, :, :, 3]) * anchor[:, 1],
            ],
            4,
        )

    def sigmoid_inverse(self, x: Tensor) -> Tensor:
        """Inverse of the sigmoid function

        Args:
            x (Tensor): the input tensor

        Returns:
            Tensor: the output tensor
        """

        # Use clamp to prevent edge cases
        return torch.log(torch.clamp(x / (1 - x), 1e-8, 1 - 1e-8))

    def encode_bounding_box(
        self,
        boxes: Tensor,
        offset_x: Tensor,
        offset_y: Tensor,
        anchor: Tensor,
        size: Tensor,
    ) -> Tensor:
        """Convert the CXCYWH-styled boxes to raw format

        Args:
            boxes (Tensor): CXCYWH-styled boxes, which is relative to the origin
            offset_x (Tensor): grid cells' top left corners' offset from the origin
            offset_y (Tensor): grid cells' top left corners' offset from the origin
            anchor (Tensor): anchor boxes
            size (Tensor): size of the corresponding feature map

        Returns:
            Tensor: bounding boxes in raw format
        """

        return torch.stack(
            [
                self.sigmoid_inverse(boxes[:, :, :, :, 0] * size - offset_x),
                self.sigmoid_inverse(boxes[:, :, :, :, 1] * size - offset_y),
                torch.log(boxes[:, :, :, :, 2] / anchor[:, 0]),
                torch.log(boxes[:, :, :, :, 3] / anchor[:, 1]),
            ],
            4,
        )
