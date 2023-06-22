"""This module provides a LightningModule for segmentation."""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from utils.image_utils import fig2png
import matplotlib.pyplot as plt


class SegmentationModel(L.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.iou = torchmetrics.JaccardIndex(num_classes=22, task='multiclass')
        self.validation_step_outputs = []
        self.training_step_outputs = []
    
    def forward(self, image):
        segmentation_mask = self.model(image)
        return segmentation_mask
    
    def _get_segmentation_mask(self, y_hat):
        # y_hat is a batch e.g. (32, 22, 224, 224)
        segmentation_mask = torch.argmax(y_hat, dim=1)
        return segmentation_mask
    
    def training_step(self, batch, batch_idx):
        image, y = batch
        y_hat = self(image)
        loss = F.cross_entropy(y_hat, y)
        self.log("train/loss", loss, on_epoch=True)
        segmentation_mask = self._get_segmentation_mask(y_hat).detach()
        self.training_step_outputs.append((segmentation_mask, y.detach()))
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, y = batch
        y_hat = self(image)
        loss = F.cross_entropy(y_hat, y)
        self.log("val/loss", loss, on_epoch=True)
        segmentation_mask = self._get_segmentation_mask(y_hat).detach()
        self.validation_step_outputs.append((segmentation_mask, y.detach()))
        return loss
    
    def on_train_epoch_end(self):
        tensorboard = self.logger.experiment
        example = self.training_step_outputs[0][0].cpu().numpy()
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(example[0, :, :])
        image = fig2png(fig)
        tensorboard.add_image("test", image, self.current_epoch, dataformats="HWC")
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        tensorboard = self.logger.experiment
        example = self.validation_step_outputs[0][0].cpu().numpy()
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(example[0, :, :])
        image = fig2png(fig)
        tensorboard.add_image("test", image, self.current_epoch, dataformats="HWC")
        self.validation_step_outputs.clear()
        
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
        


if __name__ == "__main__":
    s = SegmentationModule()