import torch
from torch import nn
from torchvision.ops import nms

from .post_process import *
from .anchors import Anchors

from .loss import FocalLoss
from .backbone import *
from .neck import *
from .head import *

backbone_dict = {
    'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
    # 'deformable_resnet18': {'models': deformable_resnet18, 'out': [64, 128, 256, 512]},
    'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
    'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
    # 'deformable_resnet50': {'models': deformable_resnet50, 'out': [256, 512, 1024, 2048]},
    'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
    'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
    # 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}
    'mobilenetv3_large': {'models': MobileNetV3}
}

detection_neck_dict = {'FPN': FPN}


class DetRegModel(nn.Module):
    def __init__(self, model_config: dict, device='cpu'):
        """
        :param model_config: 模型配置
        """
        super().__init__()
        self.device = device

        pretrained = model_config['pretrained']
        backbone = model_config['backbone']['type']
        detection_neck = model_config['neck']['type']
        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert detection_neck in detection_neck_dict, 'segmentation_head must in: {}'.format(detection_neck_dict)

        self.name = '{}_{}'.format(backbone, detection_neck)

        backbone_model = backbone_dict[backbone]['models']
        self.backbone = backbone_model(in_channels=3, pretrained=pretrained)
        if backbone == 'mobilenetv3_large':
            backbone_out_channels = self.backbone.out_channels
        else:
            backbone_out_channels = backbone_dict[backbone]['out']
        # print('backbone_out_channels: ', backbone_out_channels)
        self.detection_body = detection_neck_dict[detection_neck](backbone_out_channels,
                                                                  **model_config['neck']['args'])
        self.classificationModel = ClassificationModel(num_features_in=self.detection_body.conv_out,
                                                       num_anchors=9, num_classes=2, feature_size=128)
        self.regressionModel = RegressionModel(num_features_in=self.detection_body.conv_out,
                                               num_anchors=9, feature_size=128)
        self.anchor = Anchors(device=self.device)
        self.focalloss = FocalLoss(device=self.device)

        self.regressBoxes = BBoxTransform(device=self.device)
        self.clipBoxes = ClipBoxes()

    def forward(self, x, annotations=None):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        detection_body_out = self.detection_body(backbone_out)
        # print('detection_body_out.size(): ', detection_body_out.size())
        classification = self.classificationModel(detection_body_out)
        regression = self.regressionModel(detection_body_out)

        # print('classification.shape, regression.shape:', classification.size(), regression.size())
        # print(classification)

        anchors = self.anchor(x).to(self.device)
        # print('anchors shape:', anchors.shape)

        if self.training:
            loss = self.focalloss(classification, regression, anchors, annotations)
            return loss
        else:
            img_batch = x
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]  # 获取最大的值 [batch_size, 230400, 1]

            scores_over_thresh = (scores > 0.5)[0, :, 0]  # 设置阈值 0.5

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0, :, :], scores[0, :, 0], 0.5)  # 设置nms的iou阈值 0.5

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


if __name__ == '__main__':
    model = DetRegModel()
    print(model)

