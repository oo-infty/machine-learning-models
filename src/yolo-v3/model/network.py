from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import Sequential, Module, Upsample

from model.backbone import ResNetBackbone
from model.blocks import YoloBlock, YoloBlockStack, YoloDetector


class YoloNetworkResult(NamedTuple):
    small: Tensor
    intermediate: Tensor
    large: Tensor


class YoloNetwork(Module):
    """The main YOLO v3 network structure

    Args:
        num_box (int): the number of boxes of each grid
        num_class (int): the number of classes
    """

    def __init__(self, num_box: int, num_class: int) -> None:
        super().__init__()

        self.backbone = ResNetBackbone()
        self.stack1 = YoloBlockStack(1024, 1024, 3, 5)
        self.stack2 = YoloBlockStack(1024, 1024, 3, 5)
        self.stack3 = YoloBlockStack(512, 512, 3, 5)
        self.upsample1 = Sequential(YoloBlock(1024, 512, 3), Upsample(scale_factor=2))
        self.upsample2 = Sequential(YoloBlock(1024, 256, 3), Upsample(scale_factor=2))
        self.detector1 = YoloDetector(1024, num_box, 5 + num_class)
        self.detector2 = YoloDetector(1024, num_box, 5 + num_class)
        self.detector3 = YoloDetector(512, num_box, 5 + num_class)

    def forward(self, x: Tensor) -> YoloNetworkResult:
        """Get the detection result

        Args:
            x (Tensor): a batch of input images represented as a tensor

        Returns:
            YoloNetworkResult: detection result
        """

        # Input shape (batch, 3, 416, 416)
        backbone_res = self.backbone(x)

        # Input shape (batch, 1024, 13, 13)
        output = self.stack1(backbone_res.final)
        # Input shape (batch, 1024, 13, 13)
        res_small = self.detector1(output)
        # Output shape (batch, boxes, fields, 13, 13)

        # Input shape (batch, 1024, 13, 13)
        upsampled_output = self.upsample1(output)
        # Input shape (batch, 512, 26, 26) + (batch, 512, 26, 26)
        input = torch.cat([upsampled_output, backbone_res.intermediate2], 1)
        # Input shape (batch, 1024, 26, 26)
        output = self.stack2(input)
        # Input shape (batch, 1024, 26, 26)
        res_intermediate = self.detector2(output)
        # Output shape (batch, boxes, fields, 26, 26)

        # Input shape (batch, 1024, 26, 26)
        upsampled_output = self.upsample2(output)
        # Input shape (batch, 256, 52, 52) + (batch, 256, 52, 52)
        input = torch.cat([upsampled_output, backbone_res.intermediate1], 1)
        # Input shape (batch, 512, 52, 52)
        output = self.stack3(input)
        # Input shape (batch, 512, 52, 52)
        res_large = self.detector3(output)
        # Output shape (batch, boxes, fields, 52, 52)

        return YoloNetworkResult(
            small=res_small,
            intermediate=res_intermediate,
            large=res_large,
        )
