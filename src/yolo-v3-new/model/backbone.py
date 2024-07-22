from torch import Tensor
from torch.nn import Module, Sequential

from model.block import ConvBlock, ResidualBlock


class DarknetBackboneBlock(Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = Sequential(
            ConvBlock(in_channels, 32, 3, 1, 1),
            ConvBlock(32, 64, 3, 2, 1),
            ResidualBlock(64, num_repeats=1),
        )
        self.conv2 = Sequential(
            ConvBlock(64, 128, 3, 2, 1),
            ResidualBlock(128, num_repeats=2),
        )
        self.conv3 = Sequential(
            ConvBlock(128, 256, 3, 2, 1),
            ResidualBlock(256, num_repeats=8),
        )
        self.conv4 = Sequential(
            ConvBlock(256, 512, 3, 2, 1),
            ResidualBlock(512, num_repeats=8),
        )
        self.conv5 = Sequential(
            ConvBlock(512, 1024, 3, 2, 1),
            ResidualBlock(1024, num_repeats=4),
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        outputs.append(x)
        x = self.conv4(x)
        outputs.append(x)
        x = self.conv5(x)
        outputs.append(x)
        return outputs
