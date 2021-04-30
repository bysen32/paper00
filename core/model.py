from torch import nn
import torch
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from . import resnet
from config import TRAIN_CLASS, TRAIN_SAMPLE
import numpy as np


class MyNet(nn.Module):
    def __init__(self, head="mlp", feat_dim=512):
        super().__init__()
        in_dim = 2048
        resnet50 = models.resnet50(pretrained=True)
        layers = list(resnet50.children())[:-2]

        self.conv = nn.Sequential(*layers)
        self.avg = nn.AdaptiveAvgPool2d(1)

        if head == 'linear':
            self.head = nn.Linear(in_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, feat_dim)
            )
        else:
            raise NotImplementedError('head not suppored: {}'.format(head))

    def forward(self, images, labels, idxs=None, flag="train"):
        conv_out = self.conv(images)
        pool_out = self.avg(conv_out).squeeze()
        # features = F.normalize(self.head(pool_out), dim=1)
        # return features
        return pool_out


def get_pairs(embeddings, labels):
    # 各个样本间的特征距离
    distance_matrix = pdist(embeddings).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().reshape(-1, 1)

    batch_size = labels.shape[0]

    # 对角 下标对
    dia_inds = np.diag_indices(batch_size)

    # 标签相等bool矩阵
    lb_eqs = (labels == labels.T)

    lb_eqs[dia_inds] = False
    dist_same = distance_matrix.copy()
    dist_same[lb_eqs == False] = np.inf
    intra_idxs = np.argmin(dist_same, axis=1)

    dist_diff = distance_matrix.copy()
    lb_eqs[dia_inds] = True
    dist_diff[lb_eqs == True] = np.inf
    inter_idxs = np.argmin(dist_diff, axis=1)

    intra_pairs = np.zeros([embeddings.shape[0], 2])
    intra_labels = np.zeros([embeddings.shape[0], 2])
    inter_pairs = np.zeros([embeddings.shape[0], 2])
    inter_labels = np.zeros([embeddings.shape[0], 2])

    for i in range(embeddings.shape[0]):
        intra_pairs[i, 0] = i
        intra_pairs[i, 1] = intra_idxs[i]

        intra_labels[i, 0] = labels[i]
        intra_labels[i, 1] = labels[intra_idxs[i]]

        inter_pairs[i, 0] = i
        inter_pairs[i, 1] = inter_idxs[i]

        inter_labels[i, 0] = labels[i]
        inter_labels[i, 1] = labels[inter_idxs[i]]

    intra_labels = torch.from_numpy(intra_labels).long().cuda()
    intra_pairs = torch.from_numpy(intra_pairs).long().cuda()
    inter_labels = torch.from_numpy(inter_labels).long().cuda()
    inter_pairs = torch.from_numpy(inter_pairs).long().cuda()

    return intra_pairs, inter_pairs, intra_labels, inter_labels


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(
        2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix
