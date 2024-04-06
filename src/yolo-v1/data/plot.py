from torch import Tensor
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

from model.session import PredictionResult
import data.label

def plot_img(img: Tensor, bounding_box: list[list[int]] = [], labels: list[str] = []) -> None:
    img = draw_bounding_boxes(img, Tensor(bounding_box), labels=labels, width=3)
    fig, ax = plt.subplots()
    ax.axis(False)
    ax.imshow(img.permute(1, 2, 0).squeeze())
    plt.show()

def plot_output(img: Tensor, output: PredictionResult) -> None:
    labels = []

    for i, value in enumerate(output.classes):
        class_name = data.label.value_to_label(value.item())
        labels.append(f"{class_name} ({round(output.confidence[i].item(), 3)})")

    img = draw_bounding_boxes(
        img,
        Tensor(output.bounding_box) * 448,
        labels=labels,
        width=5,
        colors="white",
    )

    fig, ax = plt.subplots()
    ax.axis(False)
    ax.imshow(img.permute(1, 2, 0).squeeze())
    plt.show()
