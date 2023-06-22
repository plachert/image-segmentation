"""This module provides implementation of a dummy architecture for debugging."""

import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiply = nn.Parameter(torch.rand(3, 224, 224), requires_grad=True)
        self.add = nn.Parameter(torch.rand(3, 224, 224), requires_grad=True)
        
        
    def forward(self, image):
        return torch.mul(self.multiply, image) + self.add
    
    
if __name__ == "__main__":
    a = torch.rand(1, 3, 224 ,224)
    m = DummyModel()
    print(m(a).shape)