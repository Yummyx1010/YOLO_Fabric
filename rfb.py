import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class RFB(nn.Module):
    """
    Lightweight RFB: multi-branch conv with different receptive fields.
    Keeps channels same (c1->c1).
    """
    def __init__(self, c1, scale=0.1):
        super().__init__()
        self.scale = scale

        self.branch0 = nn.Sequential(
            Conv(c1, c1, k=1, s=1)
        )
        self.branch1 = nn.Sequential(
            Conv(c1, c1, k=1, s=1),
            Conv(c1, c1, k=3, s=1, p=1, g=1)
        )
        self.branch2 = nn.Sequential(
            Conv(c1, c1, k=1, s=1),
            Conv(c1, c1, k=3, s=1, p=1),
            Conv(c1, c1, k=3, s=1, p=3, d=3)  # dilation=3
        )
        self.branch3 = nn.Sequential(
            Conv(c1, c1, k=1, s=1),
            Conv(c1, c1, k=3, s=1, p=1),
            Conv(c1, c1, k=3, s=1, p=5, d=5)  # dilation=5
        )

        self.conv_linear = Conv(c1 * 4, c1, k=1, s=1)
        self.shortcut = Conv(c1, c1, k=1, s=1)
        self.act = nn.SiLU()

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b0, b1, b2, b3], dim=1)
        out = self.conv_linear(out)
        out = out * self.scale + self.shortcut(x)
        return self.act(out)
