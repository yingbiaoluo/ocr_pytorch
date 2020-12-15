import os
import sys
import yaml
import pathlib
import argparse
from pprint import pprint
from easydict import EasyDict as edict

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# add '/text_det_reg' to python PATH
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))


def parse_arg():
    parser = argparse.ArgumentParser(description="Text detection and recognition")
    parser.add_argument('--config_file', type=str, default='./config/Det_MobileNetV3Large_fpn.yaml',
                        help='configuration filename')
    args = parser.parse_args()
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    return config


if __name__ == '__main__':
    import sys
    from torch.backends import cudnn
    from detect_text.data_loader import get_dataloader
    from detect_text.models import get_model
    from detect_text.trainer import Trainer
    from detect_text.utils import get_device

    project = 'detect_text'
    sys.path.append(os.getcwd().split(project)[0] + project)

    # -------get config-------
    config = parse_arg()
    pprint(config)

    # -------get device-------
    device = get_device(config, cudnn)
    print('Using device: ', device)

    # -------get dataloader------
    train_loader = get_dataloader(config['data']['train'])
    assert train_loader is not None
    if 'validation' in config['data']:
        val_loader = get_dataloader(config['data']['validation'])
    else:
        val_loader = None

    # from data_loader.data import vis_image
    # for img, gt_anchor in train_loader:
    #     print(img.size(), gt_anchor)
    #     print(img[0].permute(1, 2, 0).shape)
    #     vis_image(image=img[0].permute(1,2,0).numpy().astype(np.uint8).copy(), anchors=gt_anchor[0].numpy())
    #     break

    # -------get model---------
    model = get_model(config['arch'], device=device)
    # print(model)

    # -------训练、验证均写在Trainer类里----------
    trainer = Trainer(config=config,
                      model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      device=device)

    # --------训练过程------------
    trainer.train()

