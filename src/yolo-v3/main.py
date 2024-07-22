import torch
from torch import Tensor
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

from model.context import YoloContext
from model.loss import LossWeight
from session.training import TrainingSession
from session.predicition import PredictionSession, PredictionResult
from data.transform import ImagePredictionTransform, DataTransform
from data.loader import CustomVOCDetection, loader
from data.label import value_to_label

DEVICE = "cuda"
NUM_BOX = 3
NUM_CLASS = 20

EPOCH = 200
LEARNING_RATE = 1e-3
LOSS_WEIGHT = LossWeight(2.5, 1, 0.5)


def train(start_epoch: int = 1):
    context: YoloContext

    if start_epoch == 1:
        context = YoloContext(
            DEVICE,
            NUM_BOX,
            NUM_CLASS,
        )
    else:
        context = torch.load("checkpoint/yolo-v3/yolo.pth")

    training_loader = loader("train", 16)
    validation_loader = loader("trainval", 16)

    session = TrainingSession(
        DEVICE,
        context,
        "checkpoint/yolo-v3",
        training_loader,
        validation_loader,
        EPOCH,
        LEARNING_RATE,
        0.5,
        LOSS_WEIGHT,
        start_epoch,
    )

    session.run()


def predict():
    context = torch.load("checkpoint/yolo-v3/yolo.pth")

    session = PredictionSession(
        DEVICE,
        context,
        0.1,
        0.3,
    )
    testing_dataset = CustomVOCDetection(
        "data",
        "val",
        ImagePredictionTransform(),
        DataTransform(),
    )
    idx = int(torch.randint(0, len(testing_dataset), [1]).item())
    image, sample, target = testing_dataset.get(idx)
    output = session.run(sample.unsqueeze(0))[0]
    plot_output(image, output)


def plot_output(img: Tensor, output: PredictionResult) -> None:
    labels = []
    boxes = []

    for box in output.boxes:
        class_name = value_to_label(int(box.class_id.item()))
        labels.append(f"{class_name} ({round(box.confidence.item(), 3)})")
        boxes.append(box.box)
        x1, y1, x2, y2 = torch.round(box.box)
        print(f"({x1}, {y1}), ({x2}, {y2}), {labels[-1]}")

    if len(boxes) != 0:
        img = draw_bounding_boxes(
            img,
            torch.stack(boxes, 0),
            labels=labels,
            width=5,
            colors="white",
        )

    fig, ax = plt.subplots()
    ax.axis(False)
    ax.imshow(img.permute(1, 2, 0).squeeze())
    plt.show()


train(1)
# predict()
