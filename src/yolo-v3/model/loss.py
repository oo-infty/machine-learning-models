from typing import NamedTuple

from torch import Tensor
from torch.nn import Module
from torch.nn import MSELoss, BCELoss

from model.network import YoloNetworkResult
from model.context import SplitTensorResult, YoloContext


class LossWeight(NamedTuple):
    """Weights of terms in the loss function

    Args:
        coord (float): the weight of the bounding box loss term
        obj (float): the weight of has-object confidence loss term
        noobj (float): the weight of no-object confidence loss term
    """

    coord: float
    obj: float
    noobj: float


class YoloLoss(Module):
    """Calculate the loss between model's output and the ground truth

    Args:
        device (str): where to run the model
        context (YoloContext): YOLO v3 model context
        weight (LossWeight): weights of the loss function
    """

    def __init__(
        self,
        device: str,
        context: YoloContext,
        weight: LossWeight,
    ) -> None:
        super().__init__()
        self.device = device

        self.context = context
        self.weight = weight

    def forward(
        self,
        input: YoloNetworkResult,
        target: YoloNetworkResult,
    ) -> Tensor:
        """Calculate loss between input and target

        Args:
            input (YoloNetworkResult): input tensors
            target (YoloNetworkResult): ground truth

        Returns:
            Tensor: the loss
        """

        i = self.context.preprocess_output(input)
        t = self.context.preprocess_output(target, encode_box=True)

        loss_small = self.forward_impl(i.small, t.small)
        loss_intermediate = self.forward_impl(i.intermediate, t.intermediate)
        loss_large = self.forward_impl(i.large, t.large)
        return loss_small + loss_intermediate + loss_large

    def forward_impl(
        self,
        input: SplitTensorResult,
        target: SplitTensorResult,
    ) -> Tensor:
        """Actual implementation of the loss function. Accept a pair of input and target
        corresponding to one feature map.

        Args:
            input (SplitTensorResult): input tensors
            target (SplitTensorResult): ground truth

        Returns:
            Tensor: the loss
        """

        loss = Tensor([0.0]).to(self.device)
        batch_size = input.boxes.shape[0]
        mse_func = MSELoss(reduction="sum")
        bce_func = BCELoss(reduction="sum")

        indicator_obj = target.confidence == 1
        indicator_noobj = target.confidence == 0

        # Calculate the loss of bounding box coordination
        loss += self.weight.coord * mse_func(
            input.boxes[indicator_obj], target.boxes[indicator_obj]
        )

        # Calculate the loss of confidence
        loss += self.weight.obj * mse_func(
            input.confidence[indicator_obj], target.confidence[indicator_obj]
        )
        loss += self.weight.noobj * mse_func(
            input.confidence[indicator_noobj], target.confidence[indicator_noobj]
        )

        # Calculate the loss of probilities of classes
        loss += self.weight.obj * mse_func(
            input.classes[indicator_obj], target.classes[indicator_obj]
        )

        return loss / batch_size
