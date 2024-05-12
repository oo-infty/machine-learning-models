from typing import NamedTuple

import torch
from torch import Tensor
from torchvision.ops import batched_nms, box_convert

from model.context import SplitTensorResult, YoloContext
from model.network import YoloNetworkResult


class BoundingBoxContext(NamedTuple):
    """The context of a bounding box including its position, confidence and category
    
    Args:
        box (Tensor): the position and size of a bounding box
        confidence (Tensor): the corresponding confidence
        class_id (Tensor): the corresponding classification result
    """

    box: Tensor
    confidence: Tensor
    class_id: Tensor

class PredictionResult(NamedTuple):
    """The result of prediction of a single sample
    
    Args:
        boxes (list[BoundingBoxContext]): all potential boxes and relevant information
    """

    boxes: list[BoundingBoxContext]

class PredictionSession:
    """YOLO v3 prediction session
    
    Args:
        device (str): where to run the model
        context (YoloContext): the model context
        confidence_threshold (float): threshold to filter boxes
        iou_threshold (float): threshold to filter overlapped boxes based on IoU
    """

    def __init__(
        self,
        device: str,
        context: YoloContext,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.device = device
        self.context = context
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def run(self, samples: Tensor) -> list[PredictionResult]:
        """Run the predicting process

        Args:
            samples (Tensor): a batch of input

        Returns:
            list[PredictionResult]
        """

        self.context.network.eval()
        samples = samples.to(self.device)
        output = self.context.network(samples)
        return self.process(output)

    def process(self, output: YoloNetworkResult) -> list[PredictionResult]:
        """Convert the raw output

        Args:
            output (YoloNetworkResult): the final result
        """

        batch_num = output.small.shape[0]
        new_output = self.context.preprocess_output(output, decode_box=True)
        res = []

        for i in range(batch_num):
            boxes = torch.cat(
                [
                    new_output.small.boxes[i].reshape(-1, 4),
                    new_output.intermediate.boxes[i].reshape(-1, 4),
                    new_output.large.boxes[i].reshape(-1, 4),
                ],
                0
            )
            confidence = torch.cat(
                [
                    new_output.small.confidence[i].reshape(-1),
                    new_output.intermediate.confidence[i].reshape(-1),
                    new_output.large.confidence[i].reshape(-1),
                ]
            )
            classes = torch.cat(
                [
                    new_output.small.classes[i].reshape(-1, self.context.num_class),
                    new_output.intermediate.classes[i].reshape(-1, self.context.num_class),
                    new_output.large.classes[i].reshape(-1, self.context.num_class),
                ],
                0
            )
            sample_output = SplitTensorResult(boxes, confidence, classes)
            res.append(self.process_impl(sample_output))

        return res

    def process_impl(self, output: SplitTensorResult) -> PredictionResult:
        """Process the output of a single sample

        Args:
            output (SplitTensorResult): raw output of a single sample

        Returns:
            PredictionResult: the final result of a single sample
        """

        class_id = output.classes.argmax(1)
        confidence_mask = output.confidence > self.confidence_threshold

        new_bounding_box = output.boxes[confidence_mask]
        new_confidence = output.confidence[confidence_mask]
        new_class_id = class_id[confidence_mask]

        indices = batched_nms(
            new_bounding_box,
            new_confidence,
            new_class_id,
            self.iou_threshold,
        )

        res = []

        for index in indices:
            box = BoundingBoxContext(
                box_convert(new_bounding_box[index] * 416, "cxcywh", "xyxy"),
                new_confidence[index],
                new_class_id[index],
            )
            res.append(box)

        return PredictionResult(res)
