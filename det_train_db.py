import os
import sys
import time
import random
import shutil
import pathlib
import argparse

from importlib import import_module
from pprint import pprint

import torch

from lib.networks import build_model
from lib.utils.general import get_logger
from lib.utils.torch_utils import set_random_seed, weight_init


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/det_train_db_config.py', help='training config')
    opt = parser.parse_args()

    cfg_path = os.path.abspath(os.path.expanduser(opt.config))
    assert os.path.isfile(cfg_path), 'ERROR: --config file does not exit'
    assert cfg_path.endswith('.py'), 'ERROR: --only py type are supported'

    module_name = os.path.basename(cfg_path)[:-3]
    cfg_dir = os.path.dirname(cfg_path)
    sys.path.insert(0, cfg_dir)
    mod = import_module(module_name)
    sys.path.pop(0)

    return mod.config


if __name__ == '__main__':
    cfg = parse_args()
    os.makedirs(cfg.train_options['checkpoint_save_dir'], exist_ok=True)
    logger = get_logger('ocr_detection', log_file=os.path.join(cfg.train_options['checkpoint_save_dir'], 'train.log'))

    # ===> print train options
    train_options = cfg.train_options
    logger.info(cfg)

    device = torch.device(train_options['device'] if torch.cuda.is_available() and ('cuda' in train_options['device'])
                          else 'cpu')
    set_random_seed(cfg['SEED'], 'cuda' in train_options['device'], deterministic=True)

    # ===> build network
    net = build_model(cfg.model)
    print(net)

    # ===> weight init
    net.apply(weight_init)

    # ===> data loader
    train_loader = 0

