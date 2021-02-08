import torch
import torch.nn.functional as F
from torch import nn

from detect_text.models.CommonModules import ConvBnRelu


class FPN(nn.Module):
    def __init__(self, backbone_out_channels, inner_channels=256):
        """
        :param backbone_out_channels: backbone输出的维度
        :param inner_channels: 内部的channel数
        """
        super().__init__()
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(backbone_out_channels[0], inner_channels, kernel_size=1)
        self.reduce_conv_c3 = ConvBnRelu(backbone_out_channels[1], inner_channels, kernel_size=1)
        self.reduce_conv_c4 = ConvBnRelu(backbone_out_channels[2], inner_channels, kernel_size=1)
        self.reduce_conv_c5 = ConvBnRelu(backbone_out_channels[3], inner_channels, kernel_size=1)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=True)
        )
        self.out_channels = self.conv_out

    def forward(self, x):
        c2, c3, c4, c5 = x
        # c2 [1, 16, 160, 160]
        # c3 [1, 24, 80, 80]
        # c4 [1, 56, 40, 40]
        # c5 [1, 480, 20, 20]
        # Top-down
        p5 = self.reduce_conv_c5(c5)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)
        # print('p2 size:', p2.size())

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([p2, p3, p4, p5], dim=1)

