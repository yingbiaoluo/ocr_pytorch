import os
import time
import shutil
import torch
import numpy as np
from detect_text.utils import AverageMeter, ProgressMeter


class Trainer:
    def __init__(self, config, model, train_loader, val_loader, device='cpu'):
        self.config = config
        self.model = model
        self.model_name: str = self.model.name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = getattr(torch.optim, self.config['optimizer']['type'])(params=model.parameters(),
                                                                                **self.config['optimizer']['args'])
        self.start_epoch = self.config['trainer']['start_epoch']
        self.end_epoch = self.config['trainer']['end_epoch']
        self.checkpoint_path = self.config['trainer']['output_dir']

        self.losses = AverageMeter(name='Loss', fmt=':.4e')
        self.cls_losses = AverageMeter(name='cls_loss', fmt=':.4e')
        self.reg_losses = AverageMeter(name='reg_loss', fmt=':.4e')

        # resume or finetune
        if self.config['trainer']['resume_checkpoint']:
            if os.path.isfile(self.config['trainer']['resume_checkpoint']):
                self._load_checkpoint(self.config['trainer']['resume_checkpoint'], resume=True)
            else:
                print('===> no checkpoint found at {}'.format(self.config['trainer']['resume_checkpoint']))

        self.model.to(self.device)

    def train(self):
        for epoch in range(self.start_epoch + 1, self.end_epoch + 1):
            self.epoch_result = self._train_epoch(epoch)
            self._on_epoch_finish()
        return 0

    def _train_epoch(self, epoch):
        self.progressMeter = ProgressMeter(
            num_batches=len(self.train_loader),
            meters=[self.losses, self.cls_losses, self.reg_losses],
            prefix="Epoch [{}]".format(epoch)
        )

        print('====>Start epoch {}'.format(epoch))
        # switch to train mode
        self.model.train()
        lr = self.optimizer.param_groups[0]['lr']

        for i, (img_batch, gt_anchors) in enumerate(self.train_loader):
            # print('input shape: ', img_batch.size(), gt_anchors.size())
            # print(torch.max(img_batch))
            classification_loss, regression_loss = self.model(img_batch.to(self.device),
                                                              gt_anchors)
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss
            self.losses.update(loss.item(), img_batch.size(0))
            self.cls_losses.update(classification_loss.item(), img_batch.size(0))
            self.reg_losses.update(regression_loss.item(), img_batch.size(0))

            if bool(loss == 0):
                continue
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

            self.optimizer.step()
            if i % self.config['trainer']['print_freq'] == 0:
                self.progressMeter.display(i)
            # if i == 20:
            #     break
        return {'train_loss': loss.item(), 'lr': lr, 'epoch': epoch}

    def _eval(self):
        print('===>Start evaluation')
        self.model.eval()

        print('Evaluation on training data:')
        for i, (img_batch, gt_anchors) in enumerate(self.train_loader):
            start_time = time.time()
            scores, labels, boxes = self.model(img_batch.to(self.device),
                                               gt_anchors)
            # print('eval result:', scores, labels, boxes)

            from detect_text.data_loader.dataset import vis_image
            # print(img_batch.size(), gt_anchors)
            vis_image(img=(img_batch[0].permute(1, 2, 0).numpy() * 255.).astype(np.uint8).copy(),
                      anchors=gt_anchors[0].numpy(), img_name='train_test', idx=i)
            vis_image(img=(img_batch[0].permute(1, 2, 0).numpy() * 255.).astype(np.uint8).copy(),
                      anchors=boxes.detach().cpu().numpy(), img_name='train_pred', idx=i)
            del scores, labels, boxes
            print('elapsed time:', time.time() - start_time)
            if i == 10:
                break

        print('Evaluation on val data:')
        for i, (img_batch, gt_anchors) in enumerate(self.val_loader):
            start_time = time.time()
            scores, labels, boxes = self.model(img_batch.to(self.device),
                                               gt_anchors)
            # print('eval result:', scores, labels, boxes)

            from detect_text.data_loader.dataset import vis_image
            # print(img_batch.size(), gt_anchors)
            vis_image(img=(img_batch[0].permute(1, 2, 0).numpy() * 255.).astype(np.uint8).copy(),
                      anchors=gt_anchors[0].numpy(), img_name='val_test', idx=i)
            vis_image(img=(img_batch[0].permute(1, 2, 0).numpy() * 255.).astype(np.uint8).copy(),
                      anchors=boxes.detach().cpu().numpy(), img_name='val_pred', idx=i)
            del scores, labels, boxes
            print('elapsed time:', time.time() - start_time)

    def _on_epoch_finish(self):
        self._eval()
        self._save_checkpoint(mode_name=self.model_name, epoch=self.epoch_result['epoch'])

    def _save_checkpoint(self, mode_name, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        file_path = os.path.join(self.checkpoint_path, '{}_model_last.pth'.format(mode_name))
        torch.save(state, file_path)

        state_infer = self.model.state_dict()
        file_path_infer = os.path.join(self.checkpoint_path, '{}_model_last_infer.pth'.format(mode_name))
        torch.save(state_infer, file_path_infer)

        if save_best:
            shutil.copy(file_path, os.path.join(self.checkpoint_dir, 'model_best.pth'))
            print('saving current best: {}'.format(file_path))
        else:
            print('saving checkpoint: {}'.format(file_path))

    def _load_checkpoint(self, checkpoint_path, resume=True):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=resume)
        if resume:
            self.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print('===> resume from checkpoint {} (epoch{})'.format(checkpoint_path, self.start_epoch))
        else:
            print('===> finetune from checkpoint: {}'.format(checkpoint_path))
