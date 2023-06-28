"""This module provides a LightningModule for segmentation."""
from __future__ import annotations

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from utils.image_utils import fig2png
from utils.image_utils import plot_segmentation

dice = torchmetrics.Dice(num_classes=21)


def dice_loss(predictions, targets, smooth=1e-6, dims=(1, 2)):
    # dims corresponding to image height and width: [B, C, H, W].
    # Intersection: |G ∩ P|. Shape: (batch_size, num_classes)
    intersection = (predictions * targets).sum(dim=dims) + smooth
    # Summation: |G| + |P|. Shape: (batch_size, num_classes).
    summation = (predictions.sum(dim=dims) + targets.sum(dim=dims)) + smooth
    metric = (2.0 * intersection) / summation
    # Compute the mean over the remaining axes (batch and classes).
    # Shape: Scalar
    total = metric.mean()
    return total


class SegmentationModel(L.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.iou = torchmetrics.JaccardIndex(num_classes=21, task='multiclass')
        self.validation_step_outputs = []
        self.training_step_outputs = []

    def forward(self, image):
        segmentation_mask = self.model(image)
        return segmentation_mask

    def _get_segmentation_mask(self, y_hat):
        # y_hat is a batch e.g. (32, 22, 224, 224)
        segmentation_mask = torch.argmax(y_hat, dim=-1)
        return segmentation_mask

    def training_step(self, batch, batch_idx):
        image, y = batch['image'], batch['mask']
        y_hat = torch.permute(self(image), (0, 2, 3, 1))
        loss = F.binary_cross_entropy_with_logits(
            y_hat, y,
        )  # + dice_loss(y_hat, y)
        self.log('train/loss', loss, on_epoch=True)
        segmentation_mask = self._get_segmentation_mask(y_hat).detach()
        self.training_step_outputs.append(
            (torch.permute(image.detach(), (0, 2, 3, 1)), segmentation_mask),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image, y = batch['image'], batch['mask']
        y_hat = torch.permute(self(image), (0, 2, 3, 1))
        loss = F.binary_cross_entropy_with_logits(
            y_hat, y,
        )  # + dice_loss(y_hat, y)
        self.log('val/loss', loss, on_epoch=True)
        segmentation_mask = self._get_segmentation_mask(y_hat).detach()
        self.validation_step_outputs.append(
            (torch.permute(image.detach(), (0, 2, 3, 1)), segmentation_mask),
        )
        return loss

    def on_train_epoch_end(self):
        tensorboard = self.logger.experiment
        image = self.training_step_outputs[0][0].cpu().numpy()
        mask = self.training_step_outputs[0][1].cpu().numpy()
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        axs = np.ravel(axs)
        for i, ax in enumerate(axs):
            plot_segmentation(image[i, ...], mask[i, ...], ax)
        image = fig2png(fig)
        tensorboard.add_image(
            'test_train', image, self.current_epoch, dataformats='HWC',
        )
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        tensorboard = self.logger.experiment
        image = np.squeeze(self.validation_step_outputs[0][0].cpu().numpy())
        mask = np.squeeze(self.validation_step_outputs[0][1].cpu().numpy())
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        axs = np.ravel(axs)
        for i, ax in enumerate(axs):
            plot_segmentation(image[i, ...], mask[i, ...], ax)
        image = fig2png(fig)
        tensorboard.add_image(
            'test_val', image, self.current_epoch, dataformats='HWC',
        )
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
