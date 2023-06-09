import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class SegmentationModel(L.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.iou = torchmetrics.JaccardIndex(num_classes=22, task='multiclass')
    
    def forward(self, image):
        segmentation_mask = self.model(image)
        return segmentation_mask 
    
    def training_step(self, batch, batch_idx):
        image, y = batch
        y_hat = self(image)
        loss = F.cross_entropy(y_hat, y)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, y = batch
        y_hat = self(image)
        loss = F.cross_entropy(y_hat, y)
        self.log("val/loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
        


if __name__ == "__main__":
    s = SegmentationModule()