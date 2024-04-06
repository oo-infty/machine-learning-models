import torch
import torchvision
from torch import Tensor
from torch.nn import Module, MSELoss

from model.context import YoloContext

class YoloLoss(Module):
    def __init__(
        self,
        device: str,
        context: YoloContext,
        weights: dict[str, float],
    ) -> None:
        super().__init__()
        self.device = device
        self.context = context
        self.weights = {k: torch.as_tensor(v).to(device) for k, v in weights.items()}
        
    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        output_classes, output_confidence, output_bounding_box = self.context.split_tensor(output)
        target_classes, target_confidence, target_bounding_box = self.context.split_tensor(target)
        converted_output_bounding_box = self.context.convert_bounding_box(output_bounding_box)
        converted_target_bounding_box = self.context.convert_bounding_box(target_bounding_box)

        mse_loss = MSELoss(reduction="sum")
        loss = Tensor([0.]).to(self.device)

        output_target_iou = torchvision.ops.box_iou(
            converted_output_bounding_box.reshape(-1, 4),
            converted_target_bounding_box.reshape(-1, 4)
        ).diag().reshape_as(output_confidence)

        shape = converted_output_bounding_box.shape
        bounding_box_id = torch \
            .arange(shape[3]) \
            .repeat([*shape[0:3], 1]) \
            .to(self.device)
        optimal_bounding_box_id = output_target_iou \
            .argmax(3) \
            .repeat_interleave(shape[3]) \
            .reshape_as(bounding_box_id)

        indicator_obj_grid = torch.all(target_confidence[:, :, :] == 1, dim=3)
        indicator_obj = ((target_confidence != 0) & (optimal_bounding_box_id == bounding_box_id))
        indicator_noobj = (indicator_obj ^ True)
        
        loss += self.weights["coord"] * mse_loss(
            output_bounding_box[indicator_obj][:, 0:2],
            target_bounding_box[indicator_obj][:, 0:2],
        )

        loss += self.weights["coord"] * mse_loss(
            torch.sqrt(output_bounding_box[indicator_obj][:, 2:4]),
            torch.sqrt(target_bounding_box[indicator_obj][:, 2:4]),
        )
        
        loss += self.weights["obj"] * mse_loss(
            output_confidence[indicator_obj],
            output_target_iou[indicator_obj],
        )

        loss += self.weights["noobj"] * (output_confidence[indicator_noobj] ** 2).sum()

        loss += self.weights["obj"] * mse_loss(
            output_classes[indicator_obj_grid],
            target_classes[indicator_obj_grid],
        )
        
        return loss / output_classes.shape[0]
