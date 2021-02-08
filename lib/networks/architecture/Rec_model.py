import torch.nn as nn
from pathlib import Path

from lib.networks.backbone.Rec_mobilenetv3 import MobileNetV3
from lib.networks.backbone.Rec_vgg import VGG
from lib.networks.neck.Rec_SequenceDecoder import SequenceDecoder
from lib.networks.head.Rec_CTCHead import CTC
from lib.utils.torch_utils import weight_init, model_info


backbone_dict = {'MobileNetV3': MobileNetV3, 'VGG': VGG}
neck_dict = {'SequenceDecoder': SequenceDecoder}
head_dict = {'CTC': CTC}


class RecModel(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, training=True):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        backbone = self.yaml['backbone']
        self.backbone = backbone_dict[backbone](ch)

        neck = self.yaml['neck']
        self.neck = neck_dict[neck](self.backbone.out_channels, hidden_size=128)

        head = self.yaml['head']
        self.head = head_dict[head](self.neck.out_channels, n_class=nc)

        self.name = f'RecModel_{backbone}_{neck}_{head}'

        if training:
            # initialize weights, bias
            self.apply(weight_init)
            self.info()
            print('')

    def forward(self, x):  # batch * 3 * 32 * 400
        x = self.backbone(x)  # batch * 512 * 1 * 100
        x = self.neck(x)  # [batch, 141, hidden_size*2]
        x = self.head(x)  # [batch, 141, 6773]
        return x

    def info(self, verbose=True):  # print model information
        model_info(self, verbose)
