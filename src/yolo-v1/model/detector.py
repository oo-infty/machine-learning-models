import torch
from torch import Tensor
from torch.nn import LeakyReLU, Module, Sequential
from torch.nn import Linear, Flatten, Unflatten

class YoloDetector(Module):
    def __init__(self, size, bounding_box, classes):
        super().__init__()

        self.size = size
        self.bounding_box = bounding_box
        self.classes = classes
        self.total = classes + 5 * bounding_box

        self.layers = Sequential(
            # (1024, 7, 7)
            Flatten(),
            # 7 * 7 * 1024
            Linear(7 * 7 * 1024, 4096),
            LeakyReLU(0.1, True),
            # 4096
            Linear(4096, size * size * self.total),
            Unflatten(1, (size, size, self.total))
        )

    def forward(self, tensor):
        return self.normalize(self.layers(tensor))

    def normalize(self, tensor: Tensor) -> Tensor:
        assert not tensor.isnan().any()
        return torch.sigmoid(tensor)
