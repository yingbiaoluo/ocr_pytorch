import cv2
import numpy as np
import torch


def get_device(config, cudnn):
    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
        print('Using GPU: ', torch.cuda.get_device_name(0))
        # cudnn
        cudnn.enabled = config.cudnn.enabled  # True设置为使用非确定性算法
        cudnn.benchmark = config.cudnn.benchmark
        # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
        # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
        print('Using cudnn.benchmark')
        cudnn.deterministic = config.cudnn.deterministic  # 卷积操作使用确定性算法，可复现
    else:
        device = torch.device("cpu:0")
        print('Warning! Using CPU.')
    return device


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def load_checkpoint(model, checkpoint_path, device, resume=False, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=resume)
    if resume:
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print('===> resume from checkpoint {} (epoch{})'.format(checkpoint_path, start_epoch))
        return model, optimizer, start_epoch
    else:
        print('===> loading weights to model from checkpoint: {}'.format(checkpoint_path))
        return model


def scale_img(image, random_coordinate=False):
    """
    对原图大小进行处理，
    :param image:
    :param random_coordinate:
    :return:
    """
    h, w, c = image.shape

    if max(h, w) > 640:
        f_scale = min(640./h, 640./w)  # scale factor
        image = cv2.resize(src=image, dsize=None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_CUBIC)
    else:
        f_scale = 1.

    h_s, w_s, c_s = image.shape  # h scaled
    image_full = 255 * np.zeros((640, 640, c), dtype=np.uint8)
    if random_coordinate:  # random coordinate
        h_random = np.random.randint(0, 640 - h + 1)
        w_random = np.random.randint(0, 640 - w + 1)
        image_full[h_random:h_random + h_s, w_random:w_random + w_s, :] = image.astype(np.uint8)
    else:
        image_full[0:h_s, 0:w_s, :] = image.astype(np.uint8)
    return image_full / 255., f_scale  # normalize

