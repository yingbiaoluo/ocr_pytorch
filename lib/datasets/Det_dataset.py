import os
import cv2
import glob
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from lib.utils.general import xyxy2xywh, xywh2xyxy


def create_dataloader(image_path, imgsz, batch_size, hyp=None, augment=False, workers=8):
    dataset = LoadImagesAndLabels(image_path, imgsz)

    batch_size = min(batch_size, len(dataset))
    nw = min([batch_size if batch_size > 1 else 0, workers])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=True,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, augment=False, hyp=None, cache_images=False):
        if os.path.isdir(path):
            self.img_files = sorted(glob.glob(path + os.sep + '*.*'))
        else:
            raise Exception('%s does not exit' % path)

        n = len(self.img_files)
        assert n > 0, 'No images found in %s' % path

        self.n = n  # number of images
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.mosaic = self.augment  # load 4 images at a time into a mosaic (only during training)

        # Define labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in
                            self.img_files]

        # Check cache
        cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Get labels
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        # Cache labels
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = tqdm(enumerate(self.label_files))
        for i, file in pbar:
            l = self.labels[i]  # label
            if l is not None and l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                # assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                self.labels[i] = l
                nf += 1  # file found
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                cache_path, nf, nm, ne, nd, n)
        if nf == 0:
            s = 'WARNING: No labels found in %s' % (os.path.dirname(file) + os.sep)
            print(s)
            assert not augment, '%s. Can not train without labels.' % s

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = image.size  # image size (width, height)
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels

                        # content = json.load(f)
                        # for sample in content['shapes']:
                        #     x1, y1 = sample['points'][0]
                        #     x2, y2 = sample['points'][1]
                        #     if sample['group_id'] is None:
                        #         cls = 0
                        #     else:
                        #         cls = int(sample['group_id'])
                        #     l.append([cls, x1, y1, x2, y2])
                        # l = np.array(l, dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = [None, None]
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        hyp = self.hyp
        if self.mosaic:
            # load mosaic
            img, labels = load_mosaic()
        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)
            print('aaa', h0, w0, h, w)

            # Letterbox
            shape = self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            print(ratio, pad)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

            nL = len(labels)  # number of labels
            if nL:
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
                labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
                labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)
        ls = []
        for l in label:
            ls.append(l)
        return torch.stack(img, 0), label


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_mosaic():
    return 0


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def vis_image(img, anchors, img_name=None, path='./vis'):
    img = img.copy()
    os.makedirs(path, exist_ok=True)  # exist_ok=True 如果目录存在 不会报错
    for anchor in anchors:
        img = cv2.rectangle(img, pt1=(int(anchor[0]), int(anchor[1])), pt2=(int(anchor[2]), int(anchor[3])),
                            color=(0, 0, 255), thickness=1)
    cv2.imwrite('{}/{}'.format(path, img_name), img)


if __name__ == '__main__':

    train_path = '../../data/detection/images/train'
    dataset = LoadImagesAndLabels(path=train_path)

    print(dataset.labels, '\n', dataset.shapes)

    for i in range(len(dataset)):
        image, label = dataset[i]
        image_name = dataset.img_files[i]
        print(image_name.split('/'))
        print('image shape: ', image.shape, image_name)
        img = image.numpy().transpose(1, 2, 0)[:, :, ::-1]
        print(img.shape)
        print(label[:, 1:5])
        label[:, [2, 4]] *= img.shape[0]  # normalized height 0-1
        label[:, [1, 3]] *= img.shape[1]  # normalized width 0-1
        print(label[:, 1:5])
        label

        vis_image(img, label[:, 1:5], img_name=image_name.split('/')[-1])

        # for l in label:
        #     print('sss', l)
        #     print((int(l[1]), int(l[2])), (int(l[3]), int(l[4])))
        #     img = cv2.rectangle(img, pt1=(int(l[1]), int(l[2])), pt2=(int(l[3]), int(l[4])), color=(0, 255, 0))
        #
        # cv2.imwrite(Path(image_name).name, img)
