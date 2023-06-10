from torchvision.datasets import VOCSegmentation
import lightning as L
import pathlib
from typing import Callable, Optional
from torch.utils.data import random_split, DataLoader
import torch.utils.data as data_utils

VOC_PATH = pathlib.Path("/home/piotr/github/image-segmentation/data/vocsegmentation") # TODO: handle paths

class VOCDatamodule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: pathlib.Path = VOC_PATH,
        input_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 32,
        num_workers: int = 2,
        ):
        super().__init__()
        self.data_dir = data_dir
        self.input_transform = input_transform if input_transform is not None else lambda x: x
        self.target_transform = target_transform if target_transform is not None else lambda x: x
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        VOCSegmentation(self.data_dir, image_set="trainval", download=True, year='2012')

    def setup(self, stage: str):
        self.train_ds = data_utils.Subset(VOCSegmentation(
            self.data_dir,
            transform=self.input_transform,
            target_transform=self.target_transform,
            image_set="train", 
            download=False, 
            year='2012',
            ), [0])
        val_org = data_utils.Subset(VOCSegmentation(
            self.data_dir,
            transform=self.input_transform,
            target_transform=self.target_transform,
            image_set="val", 
            download=False, 
            year='2012',
            ), [0])
        new_val_size = len(val_org) // 2
        test_size = len(val_org) - new_val_size
        self.val_ds, self.test_ds = random_split(val_org, [new_val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.test_batch_size, num_workers=self.num_workers)
