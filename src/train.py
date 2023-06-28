from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers
from model.architectures.unet import UNet
from model.model import SegmentationModel

from data.datamodules import VOCDatamodule


def main():
    # mean of the imagenet dataset for normalizing
    train_transform = A.Compose(
        [
            A.Resize(height=160, width=160),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    inference_transform = A.Compose(
        [
            A.Resize(height=160, width=160),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    datamodule = VOCDatamodule(
        train_transform=train_transform, inference_transform=inference_transform,
    )
    model = SegmentationModel(UNet())

    tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/')
    trainer = Trainer(max_epochs=500, accelerator='gpu', logger=tb_logger)
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
