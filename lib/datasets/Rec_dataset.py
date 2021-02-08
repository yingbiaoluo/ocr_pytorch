import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from lib.utils.general import resize_padding


def create_dataloader(image_path, label_path, imgsz, batch_size, hyp=None, augment=False, workers=8):
    dataset = Dataset_Rec(image_path, label_path, imgsz)

    batch_size = min(batch_size, len(dataset))
    nw = min([batch_size if batch_size > 1 else 0, workers])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=True,
                                             pin_memory=True,
                                             collate_fn=Dataset_Rec.collate_fn)
    return dataloader, dataset


class Dataset_Rec(Dataset):
    def __init__(self, image_path, label_path, img_size, filter_img=False, transform=None):
        super().__init__()
        self.image_path = image_path
        self.height, self.width = img_size  # height=32
        self.images, self.labels = self.get_imagelabels(label_path)
        print('original images: %s' % len(self.images))

        if filter_img:
            self.images, self.labels = self.filter(self.images, self.labels)
        print('satisfied images: %s' % len(self.images))
        self.transform = transform

    @staticmethod
    def get_imagelabels(label_path):
        with open(label_path) as f:
            content = [[a.split(' ', 1)[0], a.strip().split(' ', 1)[1]] for a in f.readlines()]
        images, labels = zip(*content)
        return images, labels

    def filter(self, imgs, labs):
        images, labels = [], []
        n = len(imgs)
        nf, ns = 0, 0
        pbar = tqdm(enumerate(imgs))
        for i, img in pbar:
            image_arr = cv2.imread(self.image_path+os.sep+img)
            h, w, c = image_arr.shape
            nf += 1
            if w / h < self.width / self.height:
                images.append(img)
                labels.append(labs[i])
                ns += 1
            pbar.desc = 'Scanning images (%g found, %g not satisfied for %g images)' % (nf, ns, n)

        return images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = cv2.imread(self.image_path + os.sep + self.images[index])  # (32, 280, 3)

        # 不足的，补充白色区域
        image = resize_padding(image, self.height, self.width)
        image = image.transpose(2, 0, 1)

        image = image.astype(np.float32) / 255.
        label = self.labels[index]
        return torch.from_numpy(image), label

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)
        ls = []
        for l in label:
            ls.append(l)
        return torch.stack(img, 0), label

