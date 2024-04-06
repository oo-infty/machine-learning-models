from torch.nn import Module, Sequential
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU
from torchvision.models import GoogLeNet_Weights, googlenet


class ReductionConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.layers = Sequential(
            Conv2d(in_channels, out_channels // 2, 1),
            BatchNorm2d(out_channels // 2),
            LeakyReLU(0.1, True),
            Conv2d(out_channels // 2, out_channels, kernel_size, padding="same"),
            BatchNorm2d(out_channels),
        )

    def forward(self, tensor):
        return self.layers(tensor)


class YoloInceptionExtractor(Module):
    def __init__(self):
        super().__init__()

        self.backbone = googlenet(
            transform_input=False,
            weights=GoogLeNet_Weights.DEFAULT
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, tensor):
        # All heights and widths below actually need to be doubled
        tensor = self.backbone.conv1(tensor)
        # (64, 224, 224)
        tensor = self.backbone.maxpool1(tensor)
        # (64, 112, 112)
        tensor = self.backbone.conv2(tensor)
        # (64, 112, 112)
        tensor = self.backbone.conv3(tensor)
        # (192, 112, 112)
        tensor = self.backbone.maxpool2(tensor)
        # (192, 56, 56)
        tensor = self.backbone.inception3a(tensor)
        # (256, 56, 56)
        tensor = self.backbone.inception3b(tensor)
        # (480, 56, 56)
        tensor = self.backbone.maxpool3(tensor)
        # (480, 28, 28)
        tensor = self.backbone.inception4a(tensor)
        # (512, 28, 28)
        tensor = self.backbone.inception4b(tensor)
        # (512, 28, 28)
        tensor = self.backbone.inception4c(tensor)
        # (512, 28, 28)
        tensor = self.backbone.inception4d(tensor)
        # (528, 28, 28)
        tensor = self.backbone.inception4e(tensor)
        # (832, 28, 28)
        tensor = self.backbone.maxpool4(tensor)
        # (832, 14, 14)
        tensor = self.backbone.inception5a(tensor)
        # (832, 14, 14)
        tensor = self.backbone.inception5b(tensor)
        # (1024, 14, 14)
        return tensor


class YoloExtractor(Module):
    def __init__(self):
        super().__init__()

        self.layers = Sequential(
            # (1024, 14, 14)
            ReductionConv2d(1024, 1024, 3),
            LeakyReLU(0.1, True),
            ReductionConv2d(1024, 1024, 3),
            LeakyReLU(0.1, True),
            Conv2d(1024, 1024, 3, padding="same"),
            BatchNorm2d(1024),
            LeakyReLU(0.1, True),
            Conv2d(1024, 1024, 3, stride=2, padding=1),
            BatchNorm2d(1024),
            LeakyReLU(0.1, True),
            # (1024, 7, 7)
        )

    def forward(self, tensor):
        return self.layers(tensor)


