import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Quest_dataset(Dataset):
    def __init__(self, images_root: str, labels_root: str, preprocess=False, scale=False, scale_shape=(640, 640), random_coordinate=False):
        super(Quest_dataset, self).__init__()
        self.image_root = images_root
        self.label_root = labels_root
        self.preprocess = preprocess
        self.scale = scale
        self.scale_shape = scale_shape
        self.images = self.get_image(self.image_root)
        self.labels = [i[:-4]+'.txt' for i in self.images]
        self.gt_anchors = self.get_gt_anchor(self.labels)
        self.random_coordinate = random_coordinate

    def get_image(self, img_root):
        images = sorted(os.listdir(img_root))
        return images

    def get_gt_anchor(self, labels):
        gt_anchors = []
        for label in labels:
            gt_anchor = np.zeros((50, 4))
            label_path = os.path.join(self.label_root, label)
            with open(label_path) as f:
                contents = f.readlines()
                for n, content in enumerate(contents):
                    a, b, c = content.split(' ', 2)
                    if int(b) == 0:
                        gt_anchor[n] = np.array([int(i) for i in a.split(',')])
            gt_anchor = gt_anchor[gt_anchor.sum(axis=1) > 0, :].astype(np.int)
            gt_anchors.append(gt_anchor)
        return gt_anchors

    def __getitem__(self, item):
        image_path = os.path.join(self.image_root, self.images[item])
        img = cv2.imread(image_path)
        # print(img.shape)
        gt_anchors = self.gt_anchors[item]
        if self.preprocess:
            img = self.preprocess_image(img)
        if self.scale:
            img, gt_anchors = self.scale_image(img, gt_anchors)

        img = img.transpose((2, 0, 1)).astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(gt_anchors)

    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
        th, img = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img

    def scale_image(self, img, gt_anchors):
        # print(img.shape)
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w, _ = img.shape

        gt_anchors_f = np.zeros_like(gt_anchors)
        # rescale image and gt_anchors
        if max(h, w) > self.scale_shape[0]:
            f_scale = min(self.scale_shape[0]/h, self.scale_shape[1]/w)
            img = cv2.resize(src=img, dsize=None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_CUBIC)
            if gt_anchors.any():
                for i in range(gt_anchors.shape[0]):
                    gt_anchors_f[i] = f_scale * gt_anchors[i]
        else:
            gt_anchors_f = gt_anchors

        if len(img.shape) == 2:
            img = img[..., np.newaxis]
        h_s, w_s, c_s = img.shape
        img_full = 255 * np.zeros((*self.scale_shape, c_s), dtype=np.uint8)
        if self.random_coordinate:
            h_random = np.random.randint(0, self.scale_shape[0]-h_s+1)
            w_random = np.random.randint(0, self.scale_shape[1]-w_s+1)
            img_full[h_random:h_s+h_random, w_random:w_s+w_random, :] = img.astype(np.uint8)
            gt_anchors_f[:, 0] = gt_anchors_f[:, 0]+w_random
            gt_anchors_f[:, 1] = gt_anchors_f[:, 1]+h_random
            gt_anchors_f[:, 2] = gt_anchors_f[:, 2]+w_random
            gt_anchors_f[:, 3] = gt_anchors_f[:, 3]+h_random
        else:
            img_full[0:h_s, 0:w_s, :] = img.astype(np.uint8)

        # normalize
        img_full = img_full / 255.
        gt_anchors_f = gt_anchors_f.astype(np.float32)
        return img_full, gt_anchors_f

    def __len__(self):
        return len(self.images)


def vis_image(img, anchors, img_name=None, idx=None):
    anchors = anchors.astype(np.int)
    for anchor in anchors:
        cv2.rectangle(img, (anchor[0], anchor[1]), (anchor[2], anchor[3]), color=(0, 0, 255), thickness=1)
    cv2.imwrite('./test_image/{}_{}.png'.format(img_name, idx), img)


if __name__ == '__main__':
    import cv2

    image_root = "/Users/luoyingbiao/Downloads/pycharm_workplace/OCR_pytorch/data/OCR_pic/images_question" \
                 "/train_images"
    label_root = "/Users/luoyingbiao/Downloads/pycharm_workplace/OCR_pytorch/data/OCR_pic/images_question" \
                 "/train_labels"
    dataset = Quest_dataset(image_root, label_root, scale=False, random_coordinate=True)

    for i in range(109, len(dataset)):
        image, gt_anchor = dataset[i]
        image_name = dataset.images[i]
        print(i)
        print('image shape: ', image.shape)
        print('gt anchor: ', gt_anchor)
        vis_image(image, gt_anchor, image_name, idx=i)
