import torch
from torch import Tensor
from torch.nn import Module, MSELoss, BCEWithLogitsLoss, CrossEntropyLoss

from model.context import YoloContext
from utils import intersection_over_union


class YoloLoss(Module):
    def __init__(self, context: YoloContext) -> None:
        super().__init__()
        self.context = context
        self.mse = MSELoss()
        self.bce = BCEWithLogitsLoss()
        self.entropy = CrossEntropyLoss()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = Tensor([0]).to(self.context.device)

        for i in range(3):
            loss += self.forward_impl(
                output[i].to(self.context.device),
                target[i].to(self.context.device),
                self.context.scaled_anchors[i],
            )
        
        return loss

    def forward_impl(
        self,
        output: Tensor,
        target: Tensor,
        anchors: Tensor,
    ) -> Tensor:
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0
        anchors = anchors.reshape(1, 3, 1, 1, 2)

        output_conf, output_box, output_class = self.context.split_tensor(output)
        target_conf, target_box, target_class = self.context.split_tensor(target)

        box_preds = torch.cat(
            [
                torch.sigmoid(output_box[..., 0:2]),
                torch.exp(output_box[..., 2:4]) * anchors,
            ],
            dim=-1,
        )

        ious = intersection_over_union(box_preds[obj], target_box[obj]).detach()
        no_object_loss = self.bce(output_conf[noobj], target_conf[noobj])

        object_loss = self.mse(
            torch.sigmoid(output_conf[obj]),
            ious * target_conf[obj],
        )

        output_box[..., 0:2] = torch.sigmoid(output_box[..., 0:2])
        target_box[..., 2:4] = torch.log(1e-16 + target_box[..., 2:4] / anchors)
        box_loss = self.mse(output_box[obj], target_box[obj])

        tc = torch.zeros_like(output_class[obj])
        tc[target_class[obj].to(torch.int)] = 1
        class_loss = self.bce(output_class[obj], tc)

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
