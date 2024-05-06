from typing import NamedTuple, Any
import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.ops import box_convert

from model.network import YoloNetworkResult
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
        self.anchor_boxes_cluster: ClusterResult | None = None

    def run(self) -> None:
        """Main train loop"""

        self.anchor_boxes_cluster = self.setup_cluster()
        boxes = self.anchor_boxes_cluster.cluster.center
        id_mapping = self.anchor_boxes_cluster.id_mapping
        if boxes is not None:
            self.context.anchor_boxes = boxes[id_mapping]

        optimizer = Adam(self.context.network.parameters(), self.learning_rate)
        scheduler = StepLR(optimizer, 3, 0.95)

        for i in range(1, self.epoch + 1):
            print(f"Epoch #{i}")
            self.training_epoch(optimizer)
            scheduler.step()

            if i % 10 == 0:
                print(f"Validation #{i / 10}")
                self.validation_epoch()
                torch.save(self.model_path, f"{self.model_path}/yolo.pth")

    def setup_cluster(self, use_cache: bool = True) -> ClusterResult:
        print("Setup cluster")
        cluster_path = f"{self.model_path}/anchor-boxes-cluster.pth"

        if use_cache and os.path.exists(cluster_path):
            print("  Use cached cluster")
            return torch.load(cluster_path)
        else:
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

            cluster_res = ClusterResult(kmeans, mapping)
            torch.save(cluster_res, cluster_path)
            return cluster_res

    def preprocess_target(
        self,
        target_list: list[dict[str, Any]],
        anchor_boxes_cluster: ClusterResult,
    ) -> YoloNetworkResult:
        """Convert target dictionaries to a tensor

        Args:
            target_list (list[dict[str, Any]]): the input target
            anchor_boxes_cluster (ClusterResult): cluster information of anchor boxes

        Returns:
            YoloNetworkResult: target in tensor format
        """

        # Initialize the result
        cluster = anchor_boxes_cluster.cluster.to(self.device)
        id_mapping = anchor_boxes_cluster.id_mapping.to(self.device)
        batch_size = len(target_list)
        num_box = self.context.num_box
        num_class = self.context.num_class
        target_tensors = []

        for size in self.context.size:
            # Size corresponds to different feature maps
            s = int(size.item())
            tensor = torch.zeros((batch_size, s, s, num_box, 5 + num_class))
            # Prevent log(0) in encode_bounding_box()
            tensor[..., 0:2] = 0.5
            tensor[..., 2:4] = 0.8
            target_tensors.append(tensor)

        for batch_index, target in enumerate(target_list):
            # Assign an anchor box type to each bounding box
            boxes = target["boxes"].reshape(-1, 4).to(self.device)
            cluster_id = cluster(boxes[:, 2:4] - boxes[:, 0:2]).to(self.device)
            anchor_box_id = id_mapping[cluster_id]
            cxcywh_boxes = box_convert(boxes / 416, "xyxy", "cxcywh")
            classes = Tensor(target["classes"]).to(self.device, torch.int)

            for k in range(3):
                # Store ground truth information to different tensors according to
                # bounding box's size and feature map's size
                size = self.context.size[k]
                current_mask = anchor_box_id // num_box == k
                current_anchor_box_id = anchor_box_id[current_mask] % num_box
                current_cxcywh_boxes = cxcywh_boxes[current_mask]
                current_classes = classes[current_mask]
                grid_idx = torch.floor(current_cxcywh_boxes[:, 0:2] * size).to(
                    self.device, torch.int
                )

                for i, (gx, gy) in enumerate(grid_idx):
                    id = current_anchor_box_id[i]
                    value = torch.zeros(5 + num_class)
                    value[0:4] = current_cxcywh_boxes[i]
                    value[4] = 1
                    value[5 + current_classes[i]] = 1
                    target_tensors[k][batch_index, gx, gy, id] = value

        return YoloNetworkResult(
            small=target_tensors[0].to(self.device),
            intermediate=target_tensors[1].to(self.device),
            large=target_tensors[2].to(self.device),
        )

    def training_epoch(self, optimizer: Optimizer) -> None:
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
