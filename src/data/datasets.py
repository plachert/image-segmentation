from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
from base_dataset import SegmentationDataset
from PIL import Image
from torchvision.datasets import VOCSegmentation

DATASETS_PATH = pathlib.Path(
    '/home/piotr/github/image-segmentation/data',
)


class VOCSegmentationDataset(SegmentationDataset):
    def __init__(
        self,
        transform: Callable | None = None,
        image_set: str = 'train',
        download: bool = False,
    ):
        super().__init__(transform)
        dataset_path = DATASETS_PATH.joinpath('VOC')
        self.voc_dataset = VOCSegmentation(
            dataset_path,
            image_set=image_set,
            download=download,
        )

    @property
    def images(self):
        return [pathlib.Path(path) for path in self.voc_dataset.images]

    @property
    def masks(self):
        return [pathlib.Path(path) for path in self.voc_dataset.masks]

    @property
    def classes(self) -> list[str]:
        return [
            'background',
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'potted plant',
            'sheep',
            'sofa',
            'train',
            'tv/monitor',
        ]

    def _load_image(self, path: pathlib.Path) -> np.ndarray:
        image = np.array(Image.open(path), dtype=np.float32)
        print(image.shape)

    def _load_mask(self, path: pathlib.Path) -> np.ndarray:
        mask = np.array(Image.open(path), dtype=np.float32)
        # mask = np.expand_dims(mask, axis=-1)
        mask_without_border = np.where(mask == 255., 0, mask)
        one_hot = np.eye(self.no_classes)[
            mask_without_border.astype(int)
        ].astype(np.float32)
        return one_hot


if __name__ == '__main__':
    ds = VOCSegmentationDataset()
