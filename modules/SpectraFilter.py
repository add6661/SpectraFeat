

import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_2d(x, norm='ortho'):
    return torch.fft.fft2(x, norm="ortho").real


def idct_2d(x, norm='ortho'):
    return torch.fft.ifft2(x, norm="ortho").real


class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels, ratio=(0.25, 0.25), isdct=True):
        super(DctSpatialInteraction, self).__init__()
        self.ratio = ratio
        self.isdct = isdct

        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
            )

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

    def forward(self, x):
        _, _, h0, w0 = x.size()

        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))

        idct = dct_2d(x, norm='ortho')
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight
        dct_ = idct_2d(dct, norm='ortho')
        return x * dct_


class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels, patch=(8, 8), ratio=(0.25, 0.25), isdct=True):
        super(DctChannelInteraction, self).__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct

        self.channel1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=32),
        )
        self.channel2x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=32),
        )
        self.relu = nn.ReLU()

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

    def forward(self, x):
        n, c, h, w = x.size()

        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight
        dct_ = idct_2d(dct, norm='ortho')

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))


class HighFrequencyPerception(nn.Module):
    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8), isdct=True):
        super(HighFrequencyPerception, self).__init__()

        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels)
        )

    def forward(self, x):
        spatial = self.spatial(x)
        channel = self.channel(x)
        return self.out(spatial + channel)


class HFP(nn.Module):
    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8), isdct=True):
        super(HFP, self).__init__()
        self.module = HighFrequencyPerception(in_channels, ratio, patch, isdct)

    def forward(self, x):
        return self.module(x)
