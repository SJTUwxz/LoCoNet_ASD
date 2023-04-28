import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvLayer(nn.Module):

    def __init__(self, cfg):
        super(ConvLayer, self).__init__()
        self.cfg = cfg
        self.s = cfg.MODEL.NUM_SPEAKERS
        self.conv2d = torch.nn.Conv2d(256, 256 * self.s, (self.s, 7), padding=(0, 3))
        # below line is speaker parallel 93.88 code
        # self.conv2d = torch.nn.Conv2d(256, 256 * self.s, (3, 7), padding=(0, 3))
        self.ln = torch.nn.LayerNorm(256)
        self.conv2d_1x1 = torch.nn.Conv2d(256, 512, (1, 1), padding=(0, 0))
        self.conv2d_1x1_2 = torch.nn.Conv2d(512, 256, (1, 1), padding=(0, 0))
        self.gelu = nn.GELU()

    def forward(self, x, b, s):

        identity = x    # b*s, t, c
        t = x.shape[1]
        c = x.shape[2]
        out = x.view(b, s, t, c)
        out = out.permute(0, 3, 1, 2)    # b, c, s, t

        out = self.conv2d(out)    # b, s*c, 1, t
        out = out.view(b, c, s, t)
        out = out.permute(0, 2, 3, 1)    # b, s, t, c
        out = self.ln(out)
        out = out.permute(0, 3, 1, 2)
        out = self.conv2d_1x1(out)
        out = self.gelu(out)
        out = self.conv2d_1x1_2(out)    # b, c, s, t

        out = out.permute(0, 2, 3, 1)    # b, s, t, c
        out = out.view(b * s, t, c)

        out += identity

        return out, b, s
