from __future__ import annotations

from typing import Callable

import lightning as L
from torch.utils.data import DataLoader


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        transform: Callable | None = None,
        train_batch_size: int = 128,
        val_batch_size: int = 32,
        test_batch_size: int = 1,
        num_workers: int = 2,
    ):
        super().__init__()
        self.transform = transform
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )
