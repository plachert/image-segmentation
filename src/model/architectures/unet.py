"""This module provides implementation of a modified U-Net architecture."""

import torch
import torch.nn as nn


class ContractionBlock(nn.Module):
    def __init__(self, input_chans=3, output_chans=64):
        super().__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(input_chans, output_chans, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(output_chans, output_chans, (3, 3), padding=1),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        
    def forward(self, image):
        copy = self.conv_relu(image)
        out = self.maxpool(copy)
        return copy, out

    
class ExpansionBlock(nn.Module):
    def __init__(self, input_chans=1024, output_chans=512):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(input_chans, output_chans, (4, 4), stride=2, padding=1)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(input_chans, output_chans, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(output_chans, output_chans, (3, 3), padding=1),
            nn.ReLU(),
        )
        
    def forward(self, copy, image):
        upsampled = self.upsample(image)
        concatenated = torch.concat((copy, upsampled), axis=1)
        out = self.conv_relu(concatenated)
        return out


class UNet(nn.Module):
    def __init__(self, n_classes=22):
        super().__init__()
        self.level1_down = ContractionBlock(input_chans=3, output_chans=64)
        self.level2_down = ContractionBlock(input_chans=64, output_chans=128)
        self.level3_down = ContractionBlock(input_chans=128, output_chans=256)
        self.level4_down = ContractionBlock(input_chans=256, output_chans=512)
        self.bottom = ContractionBlock(input_chans=512, output_chans=1024) # TODO: unused maxpool here
        self.level4_up = ExpansionBlock(input_chans=1024, output_chans=512)
        self.level3_up = ExpansionBlock(input_chans=512, output_chans=256)
        self.level2_up = ExpansionBlock(input_chans=256, output_chans=128)
        self.level1_up = ExpansionBlock(input_chans=128, output_chans=64)
        self.reduce_channels = nn.Conv2d(64, n_classes, (1, 1))
        
    def forward(self, image):
        # Contraction
        copy1, out = self.level1_down(image)
        copy2, out = self.level2_down(out)
        copy3, out = self.level3_down(out)
        copy4, out = self.level4_down(out)
        # Bottom
        out, _ = self.bottom(out)
        # Expansion
        out = self.level4_up(copy4, out)
        out = self.level3_up(copy3, out)
        out = self.level2_up(copy2, out)
        out = self.level1_up(copy1, out)
        out = self.reduce_channels(out)
        return out
    
    
if __name__ == "__main__":
    a = torch.rand(1, 3, 224 ,224)
    m = UNet()
    print(m(a).shape)