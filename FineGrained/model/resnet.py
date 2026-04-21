import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(
        self,
        arch,
        low_res,
    ):
        super().__init__()

        # backbone
        self.encoder = models.__dict__[arch]()
        self.feat_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        # modify the encoder for lower resolution
        if low_res:
            print("Low res is activated!")
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()
            self._reinit_all_layers()

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.encoder(x)
