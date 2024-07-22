import torch
from torch import Tensor

from data.dataset import Target
from model.network import YoloNetwork
from utils import iou_width_height


class YoloContext:
    def __init__(
        self,
        device: str,
        model: YoloNetwork,
        sizes: list[int],
        anchors: list[list[tuple[float, float]]],
        ignore_threshold: float,
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.sizes = sizes
        self.anchors = anchors
        self.scaled_anchors = (
            Tensor(self.anchors)
            * Tensor(self.sizes).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(self.device)
        self.num_anchors_per_scale = self.scaled_anchors.shape[1]
        self.num_anchors = 3 * self.num_anchors_per_scale
        self.ignore_threshold = ignore_threshold

    def split_tensor(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return x[..., 0], x[..., 1:5], x[..., 5:]

    def transform_target(self, targets: list[Target]) -> list[Tensor]:
        res = [[], [], []]

        for target in targets:
            res_target = self.transform_target_impl(target)

            for i in range(3):
                res[i].append(res_target[i])

        return [torch.stack(res) for res in res]

    def transform_target_impl(self, target: Target) -> list[Tensor]:
        res = [torch.zeros((self.num_anchors // 3, s, s, 6)) for s in self.sizes]

        for box_info in target:
            box = box_info["box"]
            class_label = box_info["class"]
            x, y, width, height = box

            ious = iou_width_height(
                torch.tensor(box[2:4]),
                Tensor(self.anchors[0] + self.anchors[1] + self.anchors[2]),
            )
            anchor_indices = ious.argsort(descending=True, dim=0)
            anchor_used = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                scale_size = self.sizes[scale_idx]
                gx = int(self.sizes[scale_idx] * y)
                gy = int(self.sizes[scale_idx] * x)
                anchor_taken = res[scale_idx][anchor_on_scale, gx, gy, 0]

                if anchor_taken:
                    continue

                if not anchor_used[scale_idx]:
                    value = torch.zeros(6)
                    value[0] = 1
                    value[1] = scale_size * x - gy
                    value[2] = scale_size * y - gx
                    value[3] = scale_size * width
                    value[4] = scale_size * height
                    value[5] = int(class_label)
                    res[scale_idx][anchor_on_scale, gx, gy] = value
                    anchor_used[scale_idx] = True
                elif ious[anchor_idx] > self.ignore_threshold:
                    res[scale_idx][anchor_on_scale, gx, gy, 0] = -1
        
        return res
