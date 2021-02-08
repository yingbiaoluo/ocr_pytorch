import torch.nn as nn

from lib.networks.CommonModules import ConvBNACT


class VGG(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ConvBNACT(in_channels, 64, kernel_size=3, stride=1,
                               padding=1, groups=1, act='relu', bn=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBNACT(64, 128, kernel_size=3, stride=1,
                               padding=1, groups=1, act='relu', bn=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBNACT(128, 256, kernel_size=3, stride=1,
                               padding=1, groups=1, act='relu', bn=True)
        self.conv4 = ConvBNACT(256, 256, kernel_size=3, stride=1,
                               padding=1, groups=1, act='relu', bn=False)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.conv5 = ConvBNACT(256, 512, kernel_size=3, stride=1,
                               padding=1, groups=1, act='relu', bn=True)
        self.conv6 = ConvBNACT(512, 512, kernel_size=3, stride=1,
                               padding=1, groups=1, act='relu', bn=False)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.conv7 = ConvBNACT(512, 512, kernel_size=2, stride=1,
                               padding=0, groups=1, act='relu', bn=True)
        self.out_channels = 512

    def forward(self, x):  # [batch, 3, 32, 560]
        x = self.conv1(x)
        x = self.pool1(x)  # [batch, 64, 16, 280]
        x = self.conv2(x)
        x = self.pool2(x)  # [batch, 128, 8, 140]
        x = self.conv3(x)
        x = self.conv4(x)  # [batch, 256, 8, 140]
        x = self.pool3(x)  # [batch, 256, 4, 141]
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool4(x)  # [batch, 512, 2, 142]
        x = self.conv7(x)  # [batch, 512, 1, 141]
        return x