from __future__ import annotations

import pathlib
from abc import ABC
from abc import abstractproperty
from typing import Callable

import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset, ABC):

    def __init__(self, transform: Callable | None = None) -> None:
        super().__init__()
        self.transform = transform

    @abstractproperty
    def images(self) -> list[pathlib.Path]:
        """Return paths of images."""

    @abstractproperty
    def masks(self) -> list[pathlib.Path]:
        """Return paths of masks."""

    @abstractproperty
    def classes(self) -> list[str]:
        """Return classes of objects."""

    @property
    def no_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self._load_image(self.images[idx])
        mask = self._load_mask(self.masks[idx])
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return {'image': image, 'mask': mask}

    def _load_image(self, path: pathlib.Path) -> np.ndarray:
        'Load image from path.'

    def _load_mask(self, path: pathlib.Path) -> np.ndarray:
        'Load mask from path.'
