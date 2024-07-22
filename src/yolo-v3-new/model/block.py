from torch import Tensor
from torch.nn import Module, ModuleList, Sequential
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU


class ConvBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        *,
        cbl=True,
    ) -> None:
        super().__init__()

        if cbl:
            self.layers = Sequential(
                Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                ),
                BatchNorm2d(out_channels),
                LeakyReLU(0.1),
            )
        else:
            self.layers = Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.layers(tensor)


class ResidualBlock(Module):
    def __init__(
        self,
        channels: int,
        residual: bool = True,
        num_repeats: int = 1,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.num_repeats = num_repeats
        self.layers = ModuleList()

        for _ in range(num_repeats):
            layer = Sequential(
                ConvBlock(channels, channels // 2, 1, 1, 0),
                ConvBlock(channels // 2, channels, 3, 1, 1),
            )
            self.layers.append(layer)

    def forward(self, tensor: Tensor) -> Tensor:
        for layer in self.layers:
            tensor = tensor + layer(tensor) if self.residual else layer(tensor)

        return tensor


class DetectorBlock(Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.layers = Sequential(
            ConvBlock(in_channels, 2 * in_channels, 3, 1, 1),
            ConvBlock(2 * in_channels, (num_classes + 5) * 3, 1, 1, 0, cbl=False),
        )
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        return (
            self.layers(x)
            .unflatten(1, (3, self.num_classes + 5))
            .permute(0, 1, 3, 4, 2)
        )
