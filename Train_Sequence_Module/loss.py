import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import warnings

import torch
import torch.nn as nn

class NormalizedFocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.5,
                 gamma=2.0,
                 eps=torch.finfo(torch.float).eps,
                 use_sigmoid=True,
                 stop_delimiter_grad=True,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='nfl_loss'):
        super(NormalizedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.use_sigmoid = use_sigmoid
        self.stop_delimiter_grad = stop_delimiter_grad
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.loss_name = loss_name

    def forward(self, seg_logit, seg_label, **kwargs):
        if seg_label.dim() == 3:
            seg_label = seg_label.unsqueeze(1)
        assert seg_logit.dim() == seg_label.dim() == 4
        assert seg_logit.size(0) == seg_label.size(0)
        assert seg_logit.size()[-2:] == seg_label.size()[-2:]

        def format_logit(x, dim):
            if x.dim() == dim + 1:
                if x.size(1) == 1:
                    x = x.squeeze(1)
                elif x.size(1) == 2:
                    x = x[:, 1] - x[:, 0]
            assert x.dim() == dim, \
                f'cannot format x with shape: {x.size()} ' \
                f'to match the dim {dim}'
            return x

        if seg_label.dim() == 2:
            seg_logit = format_logit(seg_logit, 2)
        elif seg_label.dim() in [3, 4]:
            seg_logit = format_logit(seg_logit, 3)
            if seg_label.dim() == 4:
                assert seg_label.size(1) == 1, \
                    f'cannot handle seg_label with shape: {seg_label.size()}'
                seg_label = seg_label.squeeze(1)
        else:
            raise NotImplementedError(
                f'cannot handle seg_label with shape: {seg_label.size()}')

        ignore_index = self.ignore_index
        valid_mask = (seg_label != ignore_index).flatten(1).any(-1)
        loss = torch.zeros_like(valid_mask).float()
        if not valid_mask.any():
            warnings.warn(f'no valid seg_label for '
                          f'ignore_index {ignore_index}')
            return loss
        seg_logit = seg_logit[valid_mask]
        seg_label = seg_label[valid_mask]
        seg_label_min = seg_label[seg_label != ignore_index].min().item()
        seg_label_max = seg_label[seg_label != ignore_index].max().item()
        if seg_label_min < 0 or seg_label_max > 1:
            raise ValueError(f'valid seg_label out of range [0, 1], got '
                             f'[{seg_label_min}, {seg_label_max}]')

        pixel_pred = torch.sigmoid(seg_logit) \
            if self.use_sigmoid else seg_logit
        pixel_weight = (seg_label != ignore_index).float()
        pixel_pred_min = pixel_pred.min().item()
        pixel_pred_max = pixel_pred.max().item()
        if pixel_pred_min < 0 or pixel_pred_max > 1:
            raise ValueError(f'pixel_pred out of range [0, 1], got '
                             f'[{pixel_pred_min}, {pixel_pred_max}]')

        alpha = torch.where(seg_label != 0, self.alpha, (1 - self.alpha))
        alpha = pixel_weight * alpha
        pt = torch.where(
            pixel_weight > 0,
            1.0 - (pixel_pred - seg_label.to(pixel_pred)).abs(),
            torch.ones_like(pixel_pred))

        beta = (1.0 - pt) ** self.gamma
        scale = pixel_weight.sum(dim=(-2, -1), keepdim=True) / \
                (beta.sum(dim=(-2, -1), keepdim=True) + self.eps)
        if self.stop_delimiter_grad:
            scale = scale.detach()
        beta = scale * beta
        pixel_loss = -alpha * beta * (pt + self.eps).clip(None, 1.0).log()
        pixel_loss = self.loss_weight * pixel_weight * pixel_loss

        loss[valid_mask] = pixel_loss.flatten(start_dim=1).sum(-1) / \
                           pixel_weight.flatten(start_dim=1).sum(-1)

        return loss