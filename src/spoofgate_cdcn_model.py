from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class CentralDifferenceConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        theta=0.7,
    ):
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        normal = self.conv(x)
        if self.theta == 0:
            return normal

        weight = self.conv.weight
        diff_weight = weight.sum(dim=(2, 3), keepdim=True)
        diff = F.conv2d(
            x,
            diff_weight,
            bias=None,
            stride=self.conv.stride,
            padding=0,
            dilation=1,
            groups=self.conv.groups,
        )
        return normal - self.theta * diff


class CDCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, theta=0.7):
        super().__init__()
        self.block = nn.Sequential(
            CentralDifferenceConv2d(in_channels, out_channels, stride=stride, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CentralDifferenceConv2d(out_channels, out_channels, stride=1, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpoofGateCDCN(nn.Module):
    def __init__(self, theta=0.7):
        super().__init__()
        self.stem = CDCNBlock(3, 32, stride=2, theta=theta)
        self.stage1 = CDCNBlock(32, 64, stride=2, theta=theta)
        self.stage2 = CDCNBlock(64, 96, stride=2, theta=theta)
        self.stage3 = CDCNBlock(96, 128, stride=2, theta=theta)
        self.map_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        live_map_logits = self.map_head(x)
        live_logit = live_map_logits.mean(dim=(2, 3))
        return live_logit, live_map_logits


def make_live_depth_prior(height, width, device):
    y = torch.linspace(-1.0, 1.0, height, device=device).view(height, 1)
    x = torch.linspace(-1.0, 1.0, width, device=device).view(1, width)
    oval = torch.exp(-((x / 0.72) ** 2 + (y / 0.95) ** 2) * 1.55)
    nose = torch.exp(-((x / 0.24) ** 2 + ((y + 0.10) / 0.34) ** 2) * 2.1)
    prior = (0.72 * oval + 0.28 * nose).clamp(0.0, 1.0)
    return prior


def make_depth_targets(labels, map_shape, device):
    batch, _, height, width = map_shape
    live_prior = make_live_depth_prior(height, width, device).view(1, 1, height, width)
    spoof_prior = torch.zeros((1, 1, height, width), device=device)
    labels = labels.float().view(batch, 1, 1, 1)
    return labels * live_prior + (1.0 - labels) * spoof_prior


def save_checkpoint(path, model, threshold, image_size, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "threshold": float(threshold),
            "image_size": int(image_size),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = SpoofGateCDCN()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.75))
    image_size = int(checkpoint.get("image_size", 224))
    return model, threshold, image_size, checkpoint
