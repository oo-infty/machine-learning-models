from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import config
from utils import load_checkpoint
from data.loader import get_loaders
from model.context import YoloContext
from model.loss import YoloLoss


class TrainSession:
    def __init__(
        self,
        context: YoloContext,
        learning_rate: float,
        weight_decay: float,
        epoch: int,
        validation_inteval: int | None,
    ) -> None:
        self.context = context
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.validation_interval = validation_inteval

    def run(self) -> None:
        optimizer = Adam(
            self.context.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        criterion = YoloLoss(self.context)
        scaler = torch.cuda.amp.GradScaler()

        train_loader, test_loader, train_eval_loader = get_loaders(
            train_csv_path=config.DATASET + "/train.csv",
            test_csv_path=config.DATASET + "/test.csv",
        )

        if config.LOAD_MODEL:
            load_checkpoint(
                config.CHECKPOINT_FILE,
                self.context.model,
                optimizer,
                self.learning_rate,
            )

        for epoch in range(1, self.epoch + 1):
            self.train_epoch(train_loader, optimizer, criterion, scaler)

            if self.validation_interval is not None:
                if epoch % self.validation_interval:
                    self.validate_epoch()

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Callable,
        scaler: GradScaler,
    ) -> None:
        self.context.model.train()
        loop = tqdm(train_loader, leave=True)
        losses = []

        for batch_idx, (sample, target) in enumerate(loop):
            sample = sample.to(self.context.device)
            target = self.context.transform_target(target)

            with torch.cuda.amp.autocast():
                out = self.context.model(sample)
                loss = criterion(out, target)

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)

    def validate_epoch(self) -> None:
        pass
        # print(f"Currently epoch {epoch}")
        # print("On Train Eval loader:")
        # print("On Train loader:")
        # check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        # if epoch > 0 and epoch % 20 == 0:
        #     check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
        #     pred_boxes, true_boxes = get_evaluation_bboxes(
        #         test_loader,
        #         model,
        #         iou_threshold=config.NMS_IOU_THRESH,
        #         anchors=config.ANCHORS,
        #         threshold=config.CONF_THRESHOLD,
        #     )
        #     mapval = mean_average_precision(
        #         pred_boxes,
        #         true_boxes,
        #         iou_threshold=config.MAP_IOU_THRESH,
        #         box_format="midpoint",
        #         num_classes=config.NUM_CLASSES,
        #     )
        #     print(f"MAP: {mapval.item()}")
        #     model.train()
