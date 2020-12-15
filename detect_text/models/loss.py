import torch
import torch.nn as nn
from .bbox_iou import bbox_overlap_ciou
from .post_process import BBoxTransform


class FocalLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(FocalLoss, self).__init__()
        self.device = device
        self.bboxTransform = BBoxTransform(device=self.device)

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j].to(self.device)

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().to(self.device))
                classification_losses.append(torch.tensor(0).float().to(self.device))
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation)  # num_anchors x num_annotations, anchors与gt框的IoU
            # print('IoU.shape:', IoU.shape)
            # for i in range(IoU.shape[0]):
            #     if torch.sum(IoU[i, :]) != 0:
            #         print(IoU[i, :])
            #     if i == 1000:
            #         break

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1, IoU_argmax表示anchors隶属于哪个gt框

            # compute the Loss for classification
            targets = torch.ones(classification.shape).to(self.device) * -1  # classification的优化目标, [160*160*9, 2]
            targets[torch.lt(IoU_max, 0.4), :] = 0  # negative索引  IoU小于0.4 直接置为0
            positive_indices = torch.ge(IoU_max, 0.5)  # IoU大于0.5, positive索引
            num_positive_anchors = positive_indices.sum()  # positive框的数量
            # print('num_positive_anchors:', num_positive_anchors)
            assigned_annotations = bbox_annotation[IoU_argmax, :]  # 给每个anchors设定regression的优化目标

            targets[positive_indices, :] = 0
            targets[positive_indices, 1] = 1  # [0, 1]为positive的优化目标

            alpha_factor = torch.ones(targets.shape).to(self.device) * alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)  # focal loss系数

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))  # 二值交叉熵

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(self.device))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the Loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]  # [x1, y1, x2, y2]
                # assigned_annotations.shape [num_positive_anchors, 4]

                anchor_widths_pi = anchor_widths[positive_indices]  # 取positive的anchors
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi  # [num_positive_anchors]
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi  # [num_positive_anchors]
                targets_dw = torch.log(gt_widths / anchor_widths_pi)  # [num_positive_anchors]
                targets_dh = torch.log(gt_heights / anchor_heights_pi)  # [num_positive_anchors]

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                # regression的优化目标 [4, num_positive_anchors]
                targets = targets.t()  # [num_positive_anchors, 4]

                targets /= torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(self.device)

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                # smooth L1 Loss
                smooth_L1_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )

                # CIoU Loss
                pred_anchors = self.bboxTransform(anchors, regressions)  # [x1, y1, x2, y2]
                pred_anchor = pred_anchors[j, :, :]
                cious = bbox_overlap_ciou(pred_anchor[positive_indices, :], assigned_annotations)
                # print('cious:', cious.shape, cious)
                cious_loss = 1. - cious

                regression_losses.append(smooth_L1_loss.mean() + cious_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().to(self.device))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU
