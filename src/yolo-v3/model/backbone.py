from typing import NamedTuple

from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU
from torchvision.models import resnet101, ResNet101_Weights


class BackboneResult(NamedTuple):
    """Intermediate and final output tensors returned by the backbone network

    Args:
        intermediate1 (Tensor): the output tensor from the last third layer
        intermediate2 (Tensor): the output tensor from the last second layer
        final (Tensor): the output tensor from the last layer
    """

    intermediate1: Tensor
    intermediate2: Tensor
    final: Tensor


class ResNetBackbone(Module):
    """YOLOv3 model's ResNet backbone neural network

    The backbone neural network extracts features from input tensors and
    returns output tensors of three different layers.
    """

    def __init__(self) -> None:
        super().__init__()

        self.backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.adapter0 = Sequential(
            Conv2d(512, 256, 1),
            BatchNorm2d(256),
            LeakyReLU(0.1),
        )
        self.adapter1 = Sequential(
            Conv2d(1024, 512, 1),
            BatchNorm2d(512),
            LeakyReLU(0.1),
        )
        self.adapter2 = Sequential(
            Conv2d(2048, 1024, 1),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
        )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, x: Tensor) -> BackboneResult:
        """Extract features from the input tensor

        Args:
            x (Tensor): a batch of input images represented as a tensor

        Returns:
            BackboneResult: extracted features from three layers
        """

        # Input shape (batch, 3, 416, 416)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Input shape (batch, 64, 104, 104)
        x = self.backbone.layer1(x)

        # Input shape (batch, 256, 104, 104)
        x = self.backbone.layer2(x)
        # Input shape (batch, 512, 52, 52)
        intermediate1 = self.adapter0(x)
        # Output shape (batch, 256, 52, 52)

        # Input shape (batch, 512, 52, 52)
        x = self.backbone.layer3(x)
        # Input shape (batch, 1024, 26, 26)
        intermediate2 = self.adapter1(x)
        # Output shape (batch, 512, 26, 26)

        # Input shape (batch, 1024, 26, 26)
        x = self.backbone.layer4(x)
        # Input shape (batch, 2048, 13, 13)
        final = self.adapter2(x)
        # Output shape (batch, 1024, 13, 13)

        return BackboneResult(intermediate1, intermediate2, final)
