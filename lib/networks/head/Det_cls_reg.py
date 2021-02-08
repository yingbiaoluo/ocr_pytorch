from torch import nn

from detect_text.models.CommonModules import ConvBnRelu


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=2, feature_size=128):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = ConvBnRelu(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = ConvBnRelu(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = ConvBnRelu(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = ConvBnRelu(feature_size, feature_size, kernel_size=3, padding=1)
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)

        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=128):
        super(RegressionModel, self).__init__()
        self.conv1 = ConvBnRelu(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = ConvBnRelu(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = ConvBnRelu(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = ConvBnRelu(feature_size, feature_size, kernel_size=3, padding=1)
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)
