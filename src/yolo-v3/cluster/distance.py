import torch
from torch import Tensor
from torch.nn import Module


class SquaredEuclideanDistance(Module):
    """Calculate the pair-wise squared euclidean distance between two sets of points"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        """Calculate the squared euclidean distance

        Args:
            lhs (Tensor): the lhs tensor
            rhs (Tensor): the rhs tensor

        Returns:
            Tensor: squared distance between two sets of points
        """

        assert lhs.shape[1:] == rhs.shape[1:]
        res = []

        for target in rhs:
            res.append(((lhs - target) ** 2).sum(dim=1))

        return torch.stack(res).transpose_(1, 0)

class IouDistance(Module):
    """Calculate the pair-wise IoU between two sets of points"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        """Calculate the IoU

        Args:
            lhs (Tensor): the lhs tensor
            rhs (Tensor): the rhs tensor

        Returns:
            Tensor: squared distance between two sets of points
        """

        assert lhs.shape[1:] == rhs.shape[1:]
        res = []

        for target in rhs:
            intersection = torch.min(lhs[:, 0], target[0]) * torch.min(lhs[:, 1], target[1])
            union = lhs[:, 0] * lhs[:, 1] + target[0] * target[1] - intersection
            res.append(intersection / union + 1e-16)

        return torch.stack(res).transpose_(1, 0)
