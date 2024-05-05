from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import StepLR

from model.context import YoloContext
from model.loss import YoloLoss, LossWeight
from cluster.kmeans import KMeans


class ClusterResult(NamedTuple):
    """K Means algorithm result

    Args:
        cluster (KMeans): K Means algorithm context
        id_mapping (Tensor): mapping from cluster ID to sorted ID
    """

    cluster: KMeans
    id_mapping: Tensor


class TrainingSession:
    """YOLO v3 training session

    Args:
        device (str): where to run the model
        context (YoloContext): the model context
        model_path (str): where to save the model
        training_loader (DataLoader): data loader of training split
        validation_loader (DataLoader): data loader of validation split
        epoch (int): number of epoches to run
        learning_rate (float): learning rate passed to optimizer
        weights (LossWeight): weight of the loss function
    """

    def __init__(
        self,
        device: str,
        context: YoloContext,
        model_path: str,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        epoch: int,
        learning_rate: float,
        weights: LossWeight,
    ) -> None:
        self.device = device
        self.context = context
        self.model_path = model_path
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.criterion = YoloLoss(self.device, self.context, weights)

    def setup_cluster(self) -> ClusterResult:
        print("Setup cluster")

        if self.context.anchor_boxes is None:
            print("  Loading bounding boxes")
            boxes = torch.cat(
                [
                    torch.cat([t["boxes"].to(self.device) for t in target])
                    for _, target in self.training_loader
                ]
            )
            boxes = boxes[:, 2:4] - boxes[:, 0:2]

            print("  Clustering bounding boxes")
            kmeans = KMeans()
            kmeans.cluster(boxes, 9)

            print("  Processing anchor boxes")

            if kmeans.center is None:
                raise ValueError("kmeans.center is None")

            anchor_boxes = kmeans.center

            area = anchor_boxes[:, 0] * anchor_boxes[:, 1]
            # A bigger bounding box is corresponded to a smaller feature map
            indices = torch.argsort(area, descending=True)
            mapping = torch.zeros(9, dtype=torch.int)
            mapping[indices] = torch.arange(9, dtype=torch.int)

            self.context.set_anchor_boxes(anchor_boxes[indices])
            return ClusterResult(kmeans, mapping)
        else:
            print("Use cached anchor boxes")
            anchor_boxes = self.context.anchor_boxes
            kmeans = KMeans(anchor_boxes)
            return ClusterResult(kmeans, torch.arange(9))
