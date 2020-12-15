import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, stride=None, size=None, ratios=None, scales=None, device='cpu'):
        super().__init__()
        self.device = device
        if stride is None:
            self.stride = 4
        if size is None:
            self.size = 80
        if ratios is None:
            self.ratios = np.array([0.05, 0.1, 0.3])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image_batch):
        image_shape = image_batch.shape[2:]
        image_shape = np.array(image_shape)  # [640, 640]

        feature_shape = image_shape // 4

        anchors = generate_anchors(base_size=self.size, ratios=self.ratios, scales=self.scales)
        shifted_anchors = shift(feature_shape, self.stride, anchors)
        return shifted_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    all_anchors = np.expand_dims(all_anchors, axis=0)
    return torch.from_numpy(all_anchors.astype(np.float32))


if __name__ == '__main__':
    image_batch = torch.randn((1, 1, 640, 640))
    anchor = Anchors()
    anchors = anchor(image_batch)

    import cv2
    for line in range(100000, anchors.shape[0]):
        img = np.ones((640, 640, 3)).astype(np.uint8)*255
        cv2.rectangle(img,
                      pt1=(int(anchors[line, 0]), int(anchors[line, 1])),
                      pt2=(int(anchors[line, 2]), int(anchors[line, 3])), color=(0,0,255))
        cv2.imshow('image', img)
        cv2.waitKey(0)
