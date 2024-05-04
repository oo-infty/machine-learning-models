from model.context import YoloContext
from session.training import TrainingSession
from model.loss import LossWeight
from data.loader import loader

ctx = YoloContext(
    "cuda",
    3,
    20,
)

training_loader = loader("train", 64)
validation_loader = loader("trainval", 2)

session = TrainingSession(
    "cuda",
    ctx,
    "output/yolo-v3",
    training_loader,
    validation_loader,
    2,
    1e-3,
    LossWeight(5, 1, 0.1),
)

kmeans = session.setup_cluster()
print(kmeans.cluster.center)
print(kmeans.id_mapping)
