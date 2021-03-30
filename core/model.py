from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet
import numpy as np
import sys
sys.path.append("../")


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 4, 200)
        )
        self.projector = projection_MLP(2048, 512)

        # API-Net struct
        # self.fc = torch.nn.Sequential(
        #     torch.nn.Dropout(p=0.5),
        #     torch.nn.Linear(2048, 200)
        # )
        # self.map1 = nn.Linear(2048 * 2, 512)
        # self.map2 = nn.Linear(512, 2048)
        # self.drop = nn.Dropout(p=0.5)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, X, targets, flag="train"):
        if flag == "train":
            batch = targets.size(0)
            images = torch.cat(X, dim=0)

            # resnet_out, rpn_feature, feature = self.pretrained_model(x)
            raw_logits, _, raw_features = self.pretrained_model(images)
            projected_features = self.projector(raw_features)

            # 取平均
            features = torch.lerp(raw_features[:batch], raw_features[batch:], 0.5)
            # features = raw_features[:batch]
            # cpu
            intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(features, targets)
            # inter_pairs_feature = features[inter_pairs[:, 0]], features[inter_pairs[:, 1]]
            inter_pairs_feature = features[inter_pairs[:, 0]], features[inter_pairs[:, 1]]
            intra_pairs_feature = features[intra_pairs[:, 0]], features[intra_pairs[:, 1]]
            # Triplet
            # RankLoss

            return raw_logits, _, raw_features, projected_features, intra_pairs_feature, inter_pairs_feature
        else:
            batch = X.size(0)
            return self.pretrained_model(X)

    def show_feature_map(self, feature_map):
        feature_map = feature_map.squeeze(0)
        feature_map = feature_map.cpu().detach().numpy()
        feature_map_num = feature_map.shape[0]
        row_num = np.ceil(np.sqrt(feature_map_num))
        plt.figure()
        for index in range(feature_map_num):
            plt.subplot(row_num, row_num, index + 1)
            plt.imshow(feature_map[index], cmap="gray")
            plt.axis("off")
            scipy.misc.imsave(str(index + 1) + ".png", feature_map[index])
        plt.show()

    def get_pairs(self, embeddings, labels):
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
        intra_pairs = np.zeros([batch_size, 2])
        intra_labels = np.zeros([batch_size, 2])

        lb_eqs[dia_inds] = True
        dist_diff = distance_matrix.copy()
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)
        inter_pairs = np.zeros([batch_size, 2])
        inter_labels = np.zeros([batch_size, 2])

        for i in range(batch_size):
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

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(
        2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix

