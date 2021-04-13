from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet
from config import TRAIN_CLASS, TRAIN_SAMPLE
import numpy as np
import sys
sys.path.append("../")

g_Features = {}
g_Labels = {}
g_InterPairs = []
g_LabelDiff = None


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 200)
        )
        self.projector = projection_MLP(2048, 512)

        # API-Net struct
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(2048, 200)
        )
        self.map1 = nn.Linear(2048 * 2, 512)
        self.map2 = nn.Linear(512, 2048)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, targets, idxs=None, flag="train"):
        global g_Features, g_Labels
        if flag == "train":
            batch = targets.size(0)
            # images = torch.cat(X, dim=0)  # 2*batch_size

            # resnet_out, rpn_feature, feature = self.pretrained_model(x)
            raw_logits, _, raw_features = self.pretrained_model(images)
            # raw_features 2*batch_size

            # map1_out = self.map1(raw_features)
            # map1_out = self.drop(map1_out)
            # projected_features = self.map2(map1_out)
            projected_features = self.projector(raw_features)  # 2048 -> 512
            # 2* batchsize

            # 融合
            # features = torch.lerp(projected_features[:batch], projected_features[batch:], 0.5)
            # features = raw_features[:batch]
            # map1_out = self.map1(raw_features)
            # map1_out = self.drop(map1_out)
            # features = self.map2(map1_out)
            # features = torch.lerp(raw_features[:batch], raw_features[batch:], 0.5)

            # for i, f in enumerate(features):
            #     g_Features[idxs[i].item()] = features[i].detach().cpu().numpy()
            #     g_Labels[idxs[i].item()] = targets[i].detach().cpu().numpy()

            intra_pairs, inter_pairs, intra_labels, inter_labels = get_pairs(
                raw_features, targets)
            features1 = torch.cat(
                [raw_features[intra_pairs[:, 0]], raw_features[inter_pairs[:, 0]]], dim=0)
            # feature1 4*batchsize
            features2 = torch.cat(
                [raw_features[intra_pairs[:, 1]], raw_features[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat(
                [intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat(
                [intra_labels[:, 1], inter_labels[:, 1]], dim=0)

            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)

            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)

            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            features1_self = torch.mul(gate1, features1)
            features2_self = torch.mul(gate2, features2)
            logit1_self = self.fc(self.drop(features1_self))
            logit2_self = self.fc(self.drop(features2_self))

            # inter_pairs_feature = features[inter_pairs[:, 0]], features[inter_pairs[:, 1]]
            # intra_pairs_feature = features[intra_pairs[:, 0]], features[intra_pairs[:, 1]]

            # return raw_logits, _, raw_features, projected_features, intra_pairs_feature, inter_pairs_feature
            return raw_logits, _, raw_features, logit1_self, logit2_self, labels1, labels2
        else:
            batch = images.size(0)
            return self.pretrained_model(images)

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


def trainend():
    global g_Features, g_Labels, g_InterPairs, g_LabelDiff
    # C20 train sample 600
    if len(g_Features) == TRAIN_SAMPLE:
        features = []
        targets = []
        g_LabelDiff = torch.zeros((TRAIN_CLASS, TRAIN_CLASS))
        for key in g_Features.keys():
            features.append(g_Features[key])
            targets.append(g_Labels[key].item())
        features = torch.tensor(features)
        targets = torch.tensor(targets)

        intra_pairs, inter_pairs, intra_labels, inter_labels = get_pairs(
            features, targets)
        g_InterPairs = inter_pairs  # idx : idx
        # 2. 计算各个类间的差异
        # 2.1 样本 vs 样本 间差异
        # 2.2 样本 vs 类 间差异
        # 2.3 类 vs 类 间差异

        # inter_labels[:, 0], inter_labels[:, 1]  # label vs label
        for inter in inter_labels:
            g_LabelDiff[inter[0], inter[1]] += 1
        # print(torch.sum(g_LabelDiff))
        # print(g_LabelDiff)
