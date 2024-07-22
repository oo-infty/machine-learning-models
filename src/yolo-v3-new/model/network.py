import torch
from torch import Tensor
from torch.nn import Module, Sequential, Upsample

from model.block import ConvBlock, ResidualBlock, DetectorBlock
from model.backbone import DarknetBackboneBlock


class YoloNetwork(Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 80,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.backbone = DarknetBackboneBlock(in_channels)

        self.conv1 = Sequential(
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ResidualBlock(1024, residual=False, num_repeats=1),
            ConvBlock(1024, 512, 1, 1, 0),
        )
        self.detector1 = DetectorBlock(512, num_classes)
        self.upsample1 = Sequential(
            ConvBlock(512, 256, 1, 1, 0),
            Upsample(scale_factor=2),
        )

        self.conv2 = Sequential(
            ConvBlock(768, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ResidualBlock(512, residual=False, num_repeats=1),
            ConvBlock(512, 256, 1, 1, 0),
        )
        self.detector2 = DetectorBlock(256, num_classes)
        self.upsample2 = Sequential(
            ConvBlock(256, 128, 1, 1, 0),
            Upsample(scale_factor=2),
        )

        self.conv3 = Sequential(
            ConvBlock(384, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            ResidualBlock(256, residual=False, num_repeats=1),
            ConvBlock(256, 128, 1, 1, 0),
        )
        self.detector3 = DetectorBlock(128, num_classes)

    def forward(self, x: Tensor) -> list[Tensor]:
        raw = self.backbone(x)
        outputs = []
        raw[2] = self.conv1(raw[2])
        outputs.append(self.detector1(raw[2]))
        raw[2] = self.upsample1(raw[2])

        raw[1] = torch.cat([raw[1], raw[2]], dim=1)
        raw[1] = self.conv2(raw[1])
        outputs.append(self.detector2(raw[1]))
        raw[1] = self.upsample2(raw[1])

        raw[0] = torch.cat([raw[0], raw[1]], dim=1)
        raw[0] = self.conv3(raw[0])
        outputs.append(self.detector3(raw[0]))

        return outputs
