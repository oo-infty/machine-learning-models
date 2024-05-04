from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import Module

from cluster.distance import SquaredEuclideanDistance


class KMeans(Module):
    """K-Means algorithm implementation

    Args:
        center (Optional[Tensor]): initial cluster center points
        dis_func (Callable)
    """

    def __init__(
        self,
        center: Optional[Tensor] = None,
        dis_func: Callable[[Tensor, Tensor], Tensor] = SquaredEuclideanDistance(),
    ) -> None:
        super().__init__()
        self.center = center
        self.num_cluster: Optional[int] = None
        self.dis_func = dis_func

        if self.center is not None:
            self.num_cluster = len(self.center)

    def set_center(self, center: Tensor) -> None:
        """Set cluster center points

        Args:
            center (Tensor): cluster center points
        """

        self.center = center
        self.num_cluster = len(self.center)

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        """Classify a batch of tensors to different clusters. Note that cluster center
        points should be found previously

        Args:
            input (Tensor): input tensors

        Returns:
            Tensor: result of classification
        """

        assert self.center is not None
        assert self.center.shape[1:] == input.shape[1:]

        return self.dis_func(input, self.center).argmin(1)

    def cluster(
        self,
        input: Tensor,
        num_cluster: int,
    ) -> None:
        """Find the cluster centers of input

        Args:
            input (Tensor): the input tensor
            num_cluster (int): the number of clusters
        """

        while True:
            res = self.cluster_impl(input, num_cluster)

            if res:
                break

    def cluster_impl(
        self,
        input: Tensor,
        num_cluster: int,
    ) -> bool:
        """Implementation of finding the cluster centers of input. Note that this
        may fail due to occurance of NaN 

        Args:
            input (Tensor): the input tensor
            num_cluster (int): the number of clusters
        """

        cluster_id = torch.randint(0, num_cluster, [input.shape[0]])
        self.num_cluster = num_cluster
        self.center = None

        while True:
            mean = []

            for i in range(num_cluster):
                points = input[cluster_id == i]

                # An empty cluster leads to NaN mean, reassigning initial cluster IDs
                # may fix this.
                if len(points) == 0:
                    print("  Restart clustering")
                    return False

                mean.append(points.mean(0, dtype=torch.float32))

            center = torch.stack(mean)

            if self.center is not None and (center == self.center).all():
                break

            self.center = center
            cluster_id = self.dis_func(input, self.center).argmin(1)

        return True
