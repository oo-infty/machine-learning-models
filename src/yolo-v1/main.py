import torch

from data.transform import ImageTrainingTransform, DataTransform
from data.loader import loader, voc_detection_dataset
from data.plot import plot_output
from model.context import BackboneType, YoloContext
from model.session import PredictionSession, TrainingSession

MODEL_PATH = "output/yolo-v1/yolo-resnet.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 7
BOUNDING_BOX = 2
CLASSES = 20
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

EPOCH = 200
LEARNING_RATE = 5e-4
BATCH_SIZE = 96
WEIGHTS = {"coord": 5, "obj": 1, "noobj": 0.5}


def main():
    # train()
    predict()


def train():
    context = YoloContext(
        DEVICE,
        SIZE,
        BOUNDING_BOX,
        CLASSES,
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        BackboneType.RESNET,
    )

    session = TrainingSession(
        DEVICE,
        context,
        MODEL_PATH,
        loader("train", BATCH_SIZE, SIZE, BOUNDING_BOX, CLASSES),
        loader("trainval", BATCH_SIZE, SIZE, BOUNDING_BOX, CLASSES),
        EPOCH,
        LEARNING_RATE,
        WEIGHTS,
    )

    session.run()


def predict():
    context = torch.load(MODEL_PATH)
    context.confidence_threshold = 0.2
    context.iou_threshold = 0.5

    session = PredictionSession(
        DEVICE,
        context,
    )

    dataset = voc_detection_dataset(
        "val",
        ImageTrainingTransform(),
        DataTransform(),
        size=SIZE,
        bounding_box=BOUNDING_BOX,
        classes=CLASSES,
    )

    idx = int(torch.randint(0, len(dataset), [1]).item())
    img, target, raw_img, raw_target = dataset.get_item(idx)

    output = session.run(img.unsqueeze(0))
    print("output =", output[0])
    print("target =", raw_target)
    plot_output(raw_img, output[0])


if __name__ == "__main__":
    main()
