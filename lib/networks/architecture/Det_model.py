from addict import Dict as AttrDict
from torch import nn

from lib.networks.backbone.Det_mobilenetv3 import MobileNetV3
from lib.networks.neck.Det_fpn import FPN
from lib.networks.head.Det_dbhead import DBHead

backbone_dict = {'MobileNetV3': MobileNetV3}
neck_dict = {'FPN': FPN}
head_dict = {'DBHead': DBHead}


class DetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert 'in_channels' in config, 'in_channels must in model config'
        backbone_type = config.backbone.pop('type')
        assert backbone_type in backbone_dict, f'backbone.type must in {backbone_dict}'
        self.backbone = backbone_dict[backbone_type](config.in_channels, **config.backbone)

        neck_type = config.neck.pop('type')
        assert neck_type in neck_dict, f'neck.type must in {neck_dict}'
        self.neck = neck_dict[neck_type](self.backbone.out_channels, **config.neck)

        head_type = config.head.pop('type')
        assert head_type in head_dict, f'head.type must in {head_dict}'
        self.head = head_dict[head_type](self.neck.out_channels, **config.head)

        self.name = f'DetModel_{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    import torch

    db_config = AttrDict(
        in_channels=3,
        backbone=AttrDict(type='MobileNetV3', model_name='large', pretrained=False),
        neck=AttrDict(type='FPN', inner_channels=256),
        head=AttrDict(type='DBHead')
    )
    x = torch.zeros(1, 3, 640, 640)
    model = DetModel(db_config)
    print(model)
    out = model(x)
    print(out.shape)
