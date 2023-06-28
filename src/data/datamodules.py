from __future__ import annotations

from torch.utils.data import random_split

from .base_datamodule import SegmentationDataModule
from .datasets import VOCSegmentationDataset


class VOCDatamodule(SegmentationDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        # download
        VOCSegmentationDataset(image_set='trainval', download=True)

    def setup(self, stage: str):
        self.train_ds = VOCSegmentationDataset(
            transform=self.transform,
            image_set='train',
            download=False,
        )
        val_org = VOCSegmentationDataset(
            transform=self.transform,
            image_set='val',
            download=False,
        )
        new_val_size = len(val_org) // 2
        test_size = len(val_org) - new_val_size
        self.val_ds, self.test_ds = random_split(
            val_org, [new_val_size, test_size],
        )
