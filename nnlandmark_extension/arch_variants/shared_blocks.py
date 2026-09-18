import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class CBAMBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        avg = self.channel_fc(self.avg_pool(x).view(b, c))
        mx = self.channel_fc(self.max_pool(x).view(b, c))
        ch_w = self.sigmoid(avg + mx).view(b, c, 1, 1, 1)
        x = x * ch_w
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True).values
        sp_w = self.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sp_w


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True, use_cbam=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        if use_cbam:
            self.se = CBAMBlock3D(out_ch)
        elif use_se:
            self.se = SEBlock3D(out_ch)
        else:
            self.se = nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return F.relu(out)


class UNetUpBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def soft_argmax_3d(heatmap):
    B, _, D, H, W = heatmap.shape
    h = heatmap.view(B, -1)
    h = torch.softmax(h, dim=-1).view(B, 1, D, H, W)
    device = heatmap.device
    d_coords = torch.linspace(0, 1, D, device=device)
    h_coords = torch.linspace(0, 1, H, device=device)
    w_coords = torch.linspace(0, 1, W, device=device)
    grid_d = d_coords.view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
    grid_h = h_coords.view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
    grid_w = w_coords.view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
    coord_d = (h * grid_d).sum(dim=[2, 3, 4])
    coord_h = (h * grid_h).sum(dim=[2, 3, 4])
    coord_w = (h * grid_w).sum(dim=[2, 3, 4])
    return torch.cat([coord_d, coord_h, coord_w], dim=1)
