import os
import time
import yaml
import logging
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# os.environ['CUDA_VISIBLE_DEVICES'] = "2, 3"

logger = logging.getLogger(__name__)

from lib.datasets.Det_dataset import create_dataloader
from lib.utils.general import (
    set_logging, check_file, increment_dir, get_latest_run
)
from lib.utils.torch_utils import select_device, init_seeds


def train(hyp, opt, device, tb_writer=None):
    logger.info(f'Hyperparameters {hyp}')  # 超参数
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory: ./det_runs/exp
    wdir = log_dir / 'weights'  # weights directory
    os.makedirs(wdir, exist_ok=True)  # exist_ok=True 如果目录存在 不会报错

    epochs, batch_size, imgsz, weights = opt.epochs, opt.batch_size, opt.img_size, opt.weights

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(1)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    train_image_path, train_label_path = data_dict['train_image'], data_dict['train_label']
    test_image_path, test_label_path = data_dict['val_image'], data_dict['val_label']


    # Trainloader 训练集
    trainloader, trainset = create_dataloader(train_image_path, imgsz, batch_size,
                                              hyp=hyp, workers=opt.workers)
    testloader, testset = create_dataloader(test_image_path, imgsz, int(batch_size/2),
                                            hyp=hyp, workers=opt.workers)
    nb = len(trainloader)  # number of batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--data', type=str, default='./config/data/Det_data.yml', help='data.yaml path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')  # DistributedDataParallel
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--logdir', type=str, default='./runsdet/', help='logging directory')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    opt = parser.parse_args()

    set_logging()  # 日志基础设置

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        log_dir = Path(ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(log_dir / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('data/hyp.finetune.yaml' if opt.weights else 'data/hyp.scratch.yaml')
        opt.hyp = '/home/lyb/ocr/text_det_reg/config/data/Det_hyp.scratch.yml'
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/det_exp0 日志目录

    logger.info(opt)  # 输出opt信息
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    device = select_device(opt.device, batch_size=opt.batch_size)  # cuda:0

    logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt.logdir)
    tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0

    train(hyp, opt, device, tb_writer)


