from __future__ import annotations

from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers
from model.architectures.unet import UNet
from model.model import SegmentationModel
from torchvision import transforms

from data.datamodule import VOCDatamodule


def replace_tensor_value_(tensor, a, b):
    tensor[tensor == a] = b
    return tensor


def main():
    # mean of the imagenet dataset for normalizing
    imagenet_mean = [0., 0., 0.]  # [0.485, 0.456, 0.406]
    imagenet_std = [1., 1., 1.]  # [0.229, 0.224, 0.225]
    input_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        ],
    )
    target_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.PILToTensor(),
            transforms.Lambda(
                lambda x: replace_tensor_value_(
                    x.squeeze(0).long(), 255, 21,
                ),
            ),
        ],

    )
    datamodule = VOCDatamodule(
        input_transform=input_transform, target_transform=target_transform,
    )
    model = SegmentationModel(UNet())

    tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/')
    trainer = Trainer(max_epochs=500, accelerator='gpu', logger=tb_logger)
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
