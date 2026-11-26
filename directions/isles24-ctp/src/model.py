"""The 2D U-Net and its loss. Identical for all arms except the input channel count."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False),
        nn.InstanceNorm2d(o, affine=True),
        nn.LeakyReLU(0.01, True),
        nn.Conv2d(o, o, 3, padding=1, bias=False),
        nn.InstanceNorm2d(o, affine=True),
        nn.LeakyReLU(0.01, True),
    )


class UNet2D(nn.Module):
    def __init__(self, in_ch: int, b: int = 32):
        super().__init__()
        ch = [b, b * 2, b * 4, b * 8]
        self.enc = nn.ModuleList(
            [block(in_ch, ch[0]), block(ch[0], ch[1]), block(ch[1], ch[2]), block(ch[2], ch[3])]
        )
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in range(3)])
        self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in range(3)])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < 3:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for i in reversed(range(3)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.out(x)


def loss_fn(logit, y):
    p = torch.sigmoid(logit)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logit, y) + dice.mean()
