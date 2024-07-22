from typing import NamedTuple, Any
import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.ops import box_convert

from model.network import YoloNetworkResult
from model.context import YoloContext
from model.loss import YoloLoss, LossWeight
from cluster.kmeans import KMeans


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
        start_epoch (int): number of epoches to be skipped
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
        ignore_threshold: float,
        weights: LossWeight,
        start_epoch: int = 1,
    ) -> None:
        self.device = device
        self.context = context
        self.model_path = model_path
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.criterion = YoloLoss(self.device, self.context, weights)
        self.ignore_threshold = ignore_threshold
        self.anchor_boxes_cluster: KMeans | None = None
        self.start_epoch = start_epoch

    def run(self) -> None:
        """Main train loop"""

        self.anchor_boxes_cluster = self.setup_cluster()
        self.context.anchor_boxes = self.anchor_boxes_cluster.center

        optimizer = SGD(self.context.network.parameters(), self.learning_rate)
        scheduler = StepLR(optimizer, 3, 0.9)

        for i in range(1, self.epoch + 1):
            if i < self.start_epoch:
                scheduler.step()
                print(f"Epoch #{i} continued")
                continue

            print(f"Epoch #{i}")
            self.training_epoch(optimizer)
            scheduler.step()
            torch.save(self.context, f"{self.model_path}/yolo.pth")

            if i % 10 == 0:
                print(f"Validation #{i / 10}")
                self.validation_epoch()

    def setup_cluster(self) -> KMeans:
        """Run clustering algorithm on all target bounding boxes

        Args:
            use_cache (bool): load cluster result from a file

        Returns:
            ClusterResult: the result
        """

        center = Tensor(
            [
                [116, 90],
                [156, 198],
                [373, 326],
                [30, 61],
                [62, 45],
                [59, 119],
                [10, 13],
                [16, 30],
                [33, 23],
            ],
        ).to(self.device)
        kmeans = KMeans(center).to(self.device)
        return kmeans

    def preprocess_target(
        self,
        target_list: list[dict[str, Any]],
        anchor_boxes_cluster: KMeans,
    ) -> YoloNetworkResult:
        """Convert target dictionaries to a tensor

        Args:
            target_list (list[dict[str, Any]]): the input target
            anchor_boxes_cluster (ClusterResult): cluster information of anchor boxes

        Returns:
            YoloNetworkResult: target in tensor format
        """

        # Initialize the result
        cluster = anchor_boxes_cluster.to(self.device)
        batch_size = len(target_list)
        num_box = self.context.num_box
        num_class = self.context.num_class
        target_tensors = []

        for size in self.context.size:
            # Size corresponds to different feature maps
            s = int(size.item())
            tensor = torch.zeros((batch_size, s, s, num_box, 5 + num_class))
            target_tensors.append(tensor)

        for batch_index, target in enumerate(target_list):
            # Assign an anchor box type to each bounding box
            boxes = target["boxes"].reshape(-1, 4).to(self.device)
            cluster_score = cluster(boxes[:, 2:4] - boxes[:, 0:2]).to(self.device)
            cluster_id = cluster_score.argmax(1)
            cxcywh_boxes = box_convert(boxes / 416, "xyxy", "cxcywh")
            classes = Tensor(target["classes"]).to(self.device, torch.int)

            for k in range(3 * num_box):
                size = self.context.size[k // 3]

                for i in range(len(boxes)):
                    if k == cluster_id[i]:
                        value = torch.zeros(5 + num_class)
                        gx = torch.floor(cxcywh_boxes[i, 0] * size).to(torch.int)
                        gy = torch.floor(cxcywh_boxes[i, 1] * size).to(torch.int)
                        value[0:4] = cxcywh_boxes[i, 0:4]
                        value[4] = 1
                        value[5 + classes[i]] = 1
                        target_tensors[k // 3][batch_index, gx, gy, k % 3] = value
                        # print(f"original box = {cxcywh_boxes[i]}, ({gx.item()}, {gy.item()}, {value[0].item()}, {value[1].item()})")
                    elif cluster_score[i, k] > self.ignore_threshold:
                        value = torch.zeros(5 + num_class)
                        gx = torch.floor(cxcywh_boxes[i, 0] * size).to(torch.int)
                        gy = torch.floor(cxcywh_boxes[i, 1] * size).to(torch.int)
                        value[4] = 0.5
                        target_tensors[k // 3][batch_index, gx, gy, k % 3] = value

            # for k in range(3):
            #     # Store ground truth information to different tensors according to
            #     # bounding box's size and feature map's size
            #     size = self.context.size[k]
            #     current_mask = anchor_box_id // num_box == k
            #     current_anchor_box_id = anchor_box_id[current_mask] % num_box
            #     current_cxcywh_boxes = cxcywh_boxes[current_mask]
            #     current_classes = classes[current_mask]
            #     grid_idx = torch.floor(current_cxcywh_boxes[:, 0:2] * size).to(
            #         self.device, torch.int
            #     )

            #     for i, (gx, gy) in enumerate(grid_idx):
            #         id = current_anchor_box_id[i]
            #         value = torch.zeros(5 + num_class)
            #         value[0:4] = current_cxcywh_boxes[i]
            #         value[4] = 1
            #         value[5 + current_classes[i]] = 1
            #         target_tensors[k][batch_index, gx, gy, id] = value

        return YoloNetworkResult(
            small=target_tensors[0].to(self.device),
            intermediate=target_tensors[1].to(self.device),
            large=target_tensors[2].to(self.device),
        )

    def training_epoch(self, optimizer: Optimizer) -> None:
        """Train loop on one epoch

        Args:
            optimizer (Optimizer): optimizer used to minimize the loss
        """

        if self.anchor_boxes_cluster is None:
            raise ValueError("self.anchor_boxes_cluster is None")

        self.context.network.train()
        batch_size = self.training_loader.batch_size or 1
        length = len(self.training_loader) * batch_size
        total_loss = 0

        for batch, (samples, targets) in enumerate(self.training_loader):
            samples = samples.to(self.device)
            targets = self.preprocess_target(targets, self.anchor_boxes_cluster)
            outputs = self.context.network(samples)
            loss = self.criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size

            if (batch + 1) % 10 == 0:
                current = (batch + 1) * batch_size
                current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
                print(
                    f"  Avg Loss: {total_loss / current:>7f}, Iteration: {current:>5d}/{length}, Learning Rate: {current_lr}"
                )

        print(f"  Training Loss: {total_loss / length}")

    def validation_epoch(self) -> None:
        """Validation loop on one epoch"""

        if self.anchor_boxes_cluster is None:
            raise ValueError("self.anchor_boxes_cluster is None")

        self.context.network.eval()
        batch_size = self.validation_loader.batch_size or 1
        length = len(self.validation_loader) * batch_size
        total_loss = 0

        with torch.no_grad():
            for _, (samples, targets) in enumerate(self.training_loader):
                samples = samples.to(self.device)
                targets = self.preprocess_target(targets, self.anchor_boxes_cluster)
                outputs = self.context.network(samples)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * batch_size

        print(f"  Validation Loss: {total_loss / length}")
