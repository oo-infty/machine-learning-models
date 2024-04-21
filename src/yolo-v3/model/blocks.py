import torch
from torch import Tensor
from torch.nn import Sequential, Module
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU


class YoloBlock(Module):
    """A basic building block for YOLO v3 model

    This block consists of a convolutional layer, a batch normalization layer
    and a leaky ReLU layer. The width and height won't be changed

    Args:
        in_channels (int): input channels of `Conv2d`
        out_channels (int): output channels of `Conv2d`
        kernel_size (int): size of kernels of `Conv2d`
        inplace (bool): whether to apply operation directly on inputs
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        self.layers = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            BatchNorm2d(out_channels),
            LeakyReLU(0.1, inplace),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Returns the output of the block

        Args:
            x (Tensor): the input tensor

        Returns:
            Tensor: the output tensor
        """

        return self.layers(x)


class YoloBlockStack(Module):
    """A series of `YoloBlock`

    Args:
        in_channels (int): input channels of `Conv2d`
        out_channels (int): output channels of `Conv2d`
        kernel_size (int): size of kernels of `Conv2d`
        num (int): the number of `YoloBlock`
        inplace (bool): whether to apply operation directly on inputs
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num: int,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        self.layers = Sequential()

        if num <= 0:
            return

        self.layers.add_module(
            f"block{1}",
            YoloBlock(
                in_channels,
                out_channels,
                kernel_size,
                inplace,
            ),
        )

        for i in range(2, num + 1):
            self.layers.add_module(
                f"block{i}",
                YoloBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    inplace,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Returns the output of the block

        Args:
            x (Tensor): the input tensor

        Returns:
            Tensor: the output tensor
        """

        return self.layers(x)


class YoloDetector(Module):
    """YOLO v3's final layers used as detector

    Args:
        channels (int): channels of the input tensor
        num_box (int): the number of boxes of each grid
        num_class (int): the number of classes
    """

    def __init__(self, channels: int, num_box: int, num_class: int) -> None:
        super().__init__()

        self.num_box = num_box
        self.num_class = num_class

        self.layers = Sequential(
            YoloBlock(channels, channels, 3),
            Conv2d(channels, num_box * (5 + num_class), 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Returns the detection result

        Args:
            x (Tensor): the input tensor

        Returns:
            Tensor: the result tensor
        """

        res = (
            self.layers(x)
            .permute((0, 2, 3, 1))
            .unflatten(3, (self.num_box, 5 + self.num_class))
        )

        res[:, :, :, :, 4:] = torch.sigmoid(res[:, :, :, :, 4:])
        # The result's shape is (batch, size, size, boxes, fields), where the last
        # component consists of three three parts:
        # - [0:4]: bounding boxes's position and size, represented as (tx, ty, tw, th)
        # - [4:5]: confidence
        # - [5:]: probabilities of classes
        return res
