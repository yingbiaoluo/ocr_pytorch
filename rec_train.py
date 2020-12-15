import os
import time
import math
import yaml
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

logger = logging.getLogger(__name__)

from lib.datasets.Rec_dataset import create_dataloader
from lib.networks.architecture.Rec_model import RecModel
from rec_test import test
from lib.utils.general import (
    set_logging, strLabelConverter, get_latest_run, check_file, increment_dir, generate_alphabets)
from lib.utils.torch_utils import select_device, init_seeds, is_parallel


def train(hyp, opt, device, tb_writer=None):
    logger.info(f'Hyperparameters {hyp}')  # 超参数
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory: ./runs/exp
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
    alphabets = generate_alphabets(data_dict['alphabets'])

    # Trainloader
    trainloader, trainset = create_dataloader(train_image_path, train_label_path,
                                              imgsz, batch_size, hyp=hyp)
    testloader, testset = create_dataloader(test_image_path, test_label_path,
                                            imgsz, int(batch_size/2), hyp=hyp)
    nb = len(trainloader)  # number of batches

    # if config.options.resume_checkpoint != '':
    #     print('resume from {}'.format(config.options.resume_checkpoint))
    #     model.load_state_dict(torch.load(config.options.resume_checkpoint, map_location='cpu'))

    # Model
    nc = len(alphabets) + 1  # 6773 number of classes
    model = RecModel(opt.cfg, ch=3, nc=nc).to(device)

    # print(model)
    pretrained = weights.endswith('.pt')
    if pretrained:
        logger.info(f'load weights from {weights}')
        model.load_state_dict(torch.load(weights, map_location=device))  # load checkpoint

    # Initialize distributed training
    # DP mode
    if cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Optimizer 优化器
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    lr_init = hyp['lr0']
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=lr_init, betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.rmsprop:
        optimizer = optim.RMSprop(pg0, lr=lr_init, momentum=hyp['momentum'])
    else:
        optimizer = optim.SGD(pg0, lr=lr_init, momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('%s Optimizer groups: %g .bias, %g conv.weight, %g other' % (
        optimizer.__module__, len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Loss function
    criterion = torch.nn.CTCLoss(reduction='sum')
    criterion = criterion.to(device)

    # Start training
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    start_epoch, best_accuracy = 0, 0.5
    conveter = strLabelConverter(alphabets)
    logger.info('Using %g dataloader workers' % trainloader.num_workers)
    logger.info('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(1, device=device)  # mean losses
        pbar = enumerate(trainloader)
        logger.info(('\n' + '%10s' * 3) % ('Epoch', 'gpu_mem', 'loss'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, (image_batch, label) in pbar:  # batch ------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)

            # # Warmup
            # if ni <= nw:
            #     xi = [0, nw]  # x interp
            #     # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            #     for j, x in enumerate(optimizer.param_groups):
            #         # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            #         x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            #         if 'momentum' in x:
            #             x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            preds = model(image_batch.to(device))  # preds shape: [batch_size, W // 4, 6773]

            bs = image_batch.shape[0]  # batch size
            preds = preds.permute(1, 0, 2)
            text, length = conveter.encode(label)
            preds_lengths = torch.tensor([preds.size(0)-1] * bs, dtype=torch.long)  # [W//4 - 1]*batch
            # preds_lengths = torch.tensor([75] * bs, dtype=torch.long)  # [64]*batch

            loss = criterion(log_probs=preds, targets=text, input_lengths=preds_lengths,
                             target_lengths=length) / bs

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mloss = (mloss * i + loss) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            if i % 10 == 0:
                s = ('%10s' * 2 + '%10.4g') % ('%g/%g' % (epoch, epochs - 1), mem, mloss.item())
                pbar.set_description(s)

            # if i == 20:
            #     break

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # Start testing
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            accuracy, edit_distance = test(model, testloader, criterion, conveter, device, max_i=500)
            logger.info('Test accuracy: %g edit distance: %g' % (accuracy, edit_distance))

        # Save model
        save = (not opt.nosave)
        if save:
            name = model.name if hasattr(model, 'name') else model.module.name
            last = wdir / f'last_{name}.pt'
            best = wdir / f'best_{name}.pt'
            logger.info('save checkpoints to %s' % wdir)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
            torch.save(msd, last)

            save_best = accuracy > best_accuracy
            logger.info('save_best: %s' % save_best)
            if save_best:
                best_accuracy = accuracy
                torch.save(msd, best)
            del msd
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training


if __name__ == "__main__":
    # python rec_train.py --data config/data/Rec_data.yml --cfg config/models/Rec_MobileNetV3_LSTM_CTC.yml
    parser = argparse.ArgumentParser(description="Train text recognition (using CRNN)")
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--data', type=str, default='./config/data/Rec_data.yml', help='data.yaml path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[32, 480], help='train,test sizes')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--rmsprop', action='store_true', help='use torch.optim.RMSprop() optimizer')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')  # DistributedDataParallel
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--logdir', type=str, default='./runs/', help='logging directory')
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
        opt.hyp = '/home/lyb/ocr/text_det_reg/config/data/Rec_hyp.scratch.yml'
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/exp1 日志目录

    logger.info(opt)  # 输出opt信息
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    device = select_device(opt.device, batch_size=opt.batch_size)  # cuda:0

    logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt.logdir)
    tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0

    train(hyp, opt, device, tb_writer)
