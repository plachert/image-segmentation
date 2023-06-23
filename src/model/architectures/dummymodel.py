"""This module provides implementation of a dummy architecture for debugging."""
from __future__ import annotations

import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiply = nn.Parameter(
            torch.rand(3, 224, 224), requires_grad=True,
        )
        self.add = nn.Parameter(torch.rand(3, 224, 224), requires_grad=True)
        self.classify = nn.Conv2d(3, 22, (1, 1))

    def forward(self, image):
        out = torch.mul(self.multiply, image) + self.add
        return self.classify(out)


if __name__ == '__main__':
    a = torch.rand(1, 3, 224, 224)
    m = DummyModel()
    print(m(a).shape)
