"""Frozen ResNet-18 feature maps."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Extractor(nn.Module):
    """Return layer2 and layer3 maps. Parameters stay frozen."""

    def __init__(self, weights: str) -> None:
        super().__init__()
        if weights == "none":
            network = resnet18(weights=None)
            self.weights_name = "none"
        elif weights == "imagenet":
            weight_enum = ResNet18_Weights.IMAGENET1K_V1
            network = resnet18(weights=weight_enum)
            self.weights_name = weight_enum.name
        else:
            raise ValueError("weights must be 'imagenet' or 'none'.")

        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "ResNet18Extractor":
        return super().train(False)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.layer1(self.stem(images))
        layer2 = self.layer2(hidden)
        layer3 = self.layer3(layer2)
        return {"layer2": layer2, "layer3": layer3}
