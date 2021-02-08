import os
import cv2
import glob
import logging
import numpy as np
from pathlib import Path

import torch
from torch.autograd import Variable
import torch.distributed as dist


class strLabelConverter(object):
    def __init__(self, alphabet_):
        """
        字符串标签转换
        """
        self.alphabet = alphabet_ + 'Ω'
        self.dict = {}
        for i, char in enumerate(self.alphabet):
            self.dict[char] = i + 1

    def encode(self, text):
        length = []
        result = []

        for item in text:
            item = item.replace(' ', '').replace('\t', '')
            length.append(len(item))
            for char in item:
                if char not in self.alphabet:
                    print('char {} not in alphabets!'.format(char))
                    char = '-'
                index = self.dict[char]
                result.append(index)
        text = result
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:  # 元素个数只有一个 number of elements
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def generate_alphabets(alphabet_path):
    """
    读取文本标签，生成字符表。
    :param alphabet_path: 文本标签.
    :return: 字符表.
    """
    with open(alphabet_path, 'r', encoding='utf-8') as file:
        alphabet = sorted(list(set(repr(''.join(file.readlines())))))
        if ' ' in alphabet:
            alphabet.remove(' ')
        alphabet = ''.join(alphabet)
    return alphabet


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


def lev_ratio(str_a, str_b):
    """
    ED距离，用来衡量单词之间的相似度
    :param str_a:
    :param str_b:
    :return:
    """
    str_a = str_a.lower()
    str_b = str_b.lower()
    matrix_ed = np.zeros((len(str_a) + 1, len(str_b) + 1), dtype=np.int)
    matrix_ed[0] = np.arange(len(str_b) + 1)
    matrix_ed[:, 0] = np.arange(len(str_a) + 1)
    for i in range(1, len(str_a) + 1):
        for j in range(1, len(str_b) + 1):
            # 表示删除a_i
            dist_1 = matrix_ed[i - 1, j] + 1
            # 表示插入b_i
            dist_2 = matrix_ed[i, j - 1] + 1
            # 表示替换b_i
            dist_3 = matrix_ed[i - 1, j - 1] + (2 if str_a[i - 1] != str_b[j - 1] else 0)
            # 取最小距离
            matrix_ed[i, j] = np.min([dist_1, dist_2, dist_3])
    # print(matrix_ed)
    levenshtein_distance = matrix_ed[-1, -1]
    sum = len(str_a) + len(str_b)
    levenshtein_ratio = (sum - levenshtein_distance) / sum
    return levenshtein_ratio


def set_logging():
    logging.basicConfig(
        format="%(asctime)s %(message)s",  # 指定输出的格式和内容, %(message)s: 打印日志信息
        level=logging.INFO)  # 设置日志级别 默认为logging.WARNING


def get_latest_run(search_dir='./runs'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file  '**'匹配所有文件、目录、子目录和子目录里的文件
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def denoise(image):
    """
    对灰色图片进行降噪（注：cv2.fastNlMeansDenoising函数处理时间较长，因此不宜采用该降噪函数）
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    ret, image = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image


def resize_padding(image, height, width):
    # resize
    h, w, c = image.shape
    image = cv2.resize(image, (0, 0), fx=height / h, fy=height / h, interpolation=cv2.INTER_LINEAR)

    # padding
    h, w, c = image.shape
    img = 255. * np.ones((height, width, c))
    if w < width:
        img[:, :w, :] = image
    else:
        r = height / h
        img = cv2.resize(image, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_LINEAR)
    return img


def padding_image_batch(image_batch, height=32, width=480):

    aspect_ratios = []
    for image in image_batch:
        h, w, c = image.shape
        aspect_ratios.append(w/h)

    max_len = int(np.ceil(32 * max(aspect_ratios)))
    pad_len = max_len if max_len > width else width
    imgs = []
    for image in image_batch:
        img = resize_padding(image, height, pad_len)
        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)

    img_batch = torch.from_numpy(np.array(imgs)) / 255.

    return img_batch.float()


logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():  # True False
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


