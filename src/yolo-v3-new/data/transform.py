import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2


def train_transforms(image_size: int) -> alb.Compose:
    scale = 1.1
    return alb.Compose(
    [
        alb.LongestMaxSize(max_size=int(image_size * scale)),
        alb.PadIfNeeded(
            min_height=int(image_size * scale),
            min_width=int(image_size * scale),
            border_mode=cv2.BORDER_CONSTANT,
            value=127,
        ),
        alb.RandomCrop(width=image_size, height=image_size),
        alb.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        alb.OneOf(
            [
                alb.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
            ],
            p=1.0,
        ),
        alb.HorizontalFlip(p=0.5),
        alb.Blur(p=0.1),
        alb.CLAHE(p=0.1),
        alb.Posterize(p=0.1),
        alb.ToGray(p=0.1),
        alb.ChannelShuffle(p=0.05),
        alb.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ],
    bbox_params=alb.BboxParams(
        format="yolo",
        min_visibility=0.4,
        label_fields=[],
    ),
)

def test_transforms(image_size: int) -> alb.Compose:
    return alb.Compose(
    [
        alb.LongestMaxSize(max_size=image_size),
        alb.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=127,
        ),
        alb.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ],
    bbox_params=alb.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

