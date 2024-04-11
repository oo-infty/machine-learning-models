from collections import namedtuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import StepLR

from model.context import YoloContext
from model.loss import YoloLoss


class TrainingSession:
    def __init__(
        self,
        device: str,
        context: YoloContext,
        model_path: str,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        epoch: int,
        learning_rate: float,
        weights: dict[str, float],
    ) -> None:
        self.device = device
        self.context = context
        self.model_path = model_path
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.criterion = YoloLoss(self.device, self.context, weights).to(self.device)

    def run(self) -> None:
        optimizer = SGD(self.context.network.parameters(), self.learning_rate, 0.9)
        scheduler = StepLR(optimizer, 3, 0.95)

        for i in range(1, self.epoch + 1):
            print(f"Epoch #{i}")
            self.training_epoch(optimizer, scheduler)
            scheduler.step()

            if i % 10 == 0:
                self.save(self.model_path)

        self.validation_epoch()

    def training_epoch(self, optimizer: Optimizer, scheduler: StepLR) -> None:
        self.context.network.train()
        batch_size = self.training_loader.batch_size or 1
        length = len(self.training_loader) * batch_size
        total_loss = 0

        for batch, (samples, targets) in enumerate(self.training_loader):
            samples = samples.to(self.device)
            targets = targets.to(self.device)
            outputs = self.context.network(samples)
            loss = self.criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size

            if batch % 10 == 0:
                current = (batch + 1) * batch_size
                current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
                print(
                    f"Avg Loss: {total_loss / current:>7f}, Iteration: {current:>5d}/{length:>5d}, Learning Rate: {current_lr}"
                )

        print(f"Training Loss: {total_loss / length}")

    def validation_epoch(self) -> None:
        self.context.network.eval()
        batch_size = self.validation_loader.batch_size or 1
        length = len(self.validation_loader) * batch_size
        total_loss = 0

        with torch.no_grad():
            for batch, (samples, targets) in enumerate(self.training_loader):
                samples = samples.to(self.device)
                targets = targets.to(self.device)
                outputs = self.context.network(samples)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * batch_size

        print(f"Validation Loss: {total_loss / length}")

    def save(self, path: str):
        torch.save(self.context, path)
        print(f"Save model to {path}")


PredictionResult = namedtuple(
    "PredictionResult",
    ["bounding_box", "classes", "confidence"],
)


class PredictionSession:
    def __init__(
        self,
        device: str,
        context: YoloContext,
    ) -> None:
        self.device = device
        self.context = context

    def run(self, samples: Tensor) -> list[PredictionResult]:
        self.context.network.eval()
        samples = samples.to(self.device)
        output = self.context.network(samples)
        return self.process(output)

    def process(self, output: Tensor) -> list[PredictionResult]:
        batch_num = output.shape[0]
        classes, confidence, bounding_box = self.context.split_tensor(output)
        bounding_box = self.context.convert_bounding_box(bounding_box)
        class_confidence, classes = self.context.select_class(classes, confidence)

        res = self.context.filter_bounding_box(
            bounding_box,
            class_confidence,
            classes,
        )

        batch_id = res[3]
        predition = []

        for i in range(batch_num):
            predition.append(
                PredictionResult(
                    res[0][batch_id == i],  # Bounding boxes
                    res[1][batch_id == i],  # Classes
                    res[2][batch_id == i],  # Class confidence
                )
            )

        return predition
