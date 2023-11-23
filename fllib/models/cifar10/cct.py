import torch.nn as nn

from fllib.models.backbones.cctnets import cct_2_3x2_32


class CCTNet10(nn.Module):
    def __init__(self):
        super(CCTNet10, self).__init__()
        self.model = cct_2_3x2_32()

    def forward(self, x):
        return self.model(x)
