"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # mask 做什么的？ 0,1 排除 i=j 情况
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            # 对角阵
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # labels 转化为列向量 为torch.eq做铺垫
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            """
            仅仅给定标签数据： - 依据标签数据生成 mask
            mask: 二维数组，判断 i,j 是否为同类
            """
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # 视图数量
        # ubind 维数-1，变tuple
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        '''
        ---------------------------------------------------------------
        以上都为初始化操作
        '''
        # compute logits : i-j matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(
            anchor_dot_contrast, dim=1, keepdim=True)  # cos相似度最大值
        # logits = anchor_dot_contrast - logits_max.detach()  # 每个元素 - 列cos_sim最大值
        logits = anchor_dot_contrast
        '''
        logits 每个元素-行最大值(1) 为什么要-1？
        '''

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases 除了自己全都有
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # 同类与不同类的映射表
        '''
        logtis_mask 对角为0其余为1 区分自己和别人
        mask 区分同类和异类
        mask * logits_mask 逐元素乘 区分同类为1 异类为0 ii为0
        此时mask 代表 同类为1，自身、异类为0
        '''

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        '''
        if i == j 0
        else exp_logits[i,j] = cossin_sim[i,j] - 1
        '''
        # ''' exp_logits 对比损失中的分母 '''
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # ''' exp_logits.sum(1) 行和 q^ {\sum^{i*j}} '''

        # 逐行累加

        # compute mean of log-likelihood over positive 正例均值
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # 有必要view?
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()

        return loss
