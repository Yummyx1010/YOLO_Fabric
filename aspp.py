import torch
import torch.nn as nn
import torch.nn.functional as F

# 用 Ultralytics 的 Conv（带 BN+SiLU），更稳定
from ultralytics.nn.modules.conv import Conv


class ASPP(nn.Module):
    """
    Lightweight ASPP for YOLOv8 backbone tail.
    Keep channels same: c -> c
    """
    def __init__(self, c, rates=(3, 5, 7)):
        super().__init__()
        c = int(c)
        r1, r2, r3 = rates

        # 1x1
        self.b0 = Conv(c, c, k=1, s=1)

        # 3x3 dilated conv branches
        self.b1 = nn.Sequential(
            Conv(c, c, k=1, s=1),
            Conv(c, c, k=3, s=1, p=r1, d=r1),
        )
        self.b2 = nn.Sequential(
            Conv(c, c, k=1, s=1),
            Conv(c, c, k=3, s=1, p=r2, d=r2),
        )
        self.b3 = nn.Sequential(
            Conv(c, c, k=1, s=1),
            Conv(c, c, k=3, s=1, p=r3, d=r3),
        )

        # global pooling branch
        self.bg = Conv(c, c, k=1, s=1)

        # fuse
        self.fuse = Conv(c * 5, c, k=1, s=1)

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)

        # global context
        g = F.adaptive_avg_pool2d(x, 1)
        g = self.bg(g)
        g = F.interpolate(g, size=x.shape[-2:], mode="nearest")

        out = torch.cat([b0, b1, b2, b3, g], dim=1)
        return self.fuse(out)
