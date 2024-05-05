from model.context import YoloContext
from session.training import TrainingSession
from model.loss import LossWeight
from data.loader import loader

ctx = YoloContext(
    "cuda",
    3,
    20,
)

training_loader = loader("train", 32)
validation_loader = loader("trainval", 32)

session = TrainingSession(
    "cuda",
    ctx,
    "output/yolo-v3",
    training_loader,
    validation_loader,
    2,
    1e-5,
    LossWeight(5, 1, 0.1),
)

session.run()
