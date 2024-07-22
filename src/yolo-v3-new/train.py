import config
import torch

from model.network import YoloNetwork
from model.context import YoloContext

from session.train import TrainSession
import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def main():
    model = YoloNetwork(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    ctx = YoloContext(
        device=config.DEVICE,
        model=model,
        sizes=config.S,
        anchors=config.ANCHORS,
        ignore_threshold=0.5,
    )

    session = TrainSession(
        context=ctx,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        epoch=config.NUM_EPOCHS,
        validation_inteval=None,
    )

    session.run()


if __name__ == "__main__":
    main()
