from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from . import resnet
from config import TRAIN_CLASS, TRAIN_SAMPLE
import numpy as np

g_Features = {}
g_Labels = {}
g_InterPairs = []
g_LabelDiff = None


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.07
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 200)
        )
        self.projector = projection_MLP(2048, 512)

        # API-Net struct
        # self.fc = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Linear(2048, 200)
        # )
        self.map1 = nn.Linear(2048 * 2, 512)
        self.map2 = nn.Linear(512, 2048)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, labels, idxs=None, flag="train"):
        global g_Features, g_Labels
        if flag == "train":
            BS = labels.size(0)
            raw_logits, _, features_raw = self.pretrained_model(images)
            # 特征归一化 便于计算余弦相似度
            features_raw = F.normalize(features_raw, dim=1)

            # motivation 负样本与不够负的问题
            # 使用x^other作为负样本，将特征的关注区域产生分歧
            # 使用特征点乘 计算mutual向量 [i,j] = fi*fj
            mutual_martix = torch.einsum(
                'ik,jk->ijk', [features_raw, features_raw])
            # gate_matrix = torch.einsum('->ij',[features_raw, mutual_martix])
            matrix_gate_i = torch.einsum(
                'ik,ijk->ijk', [features_raw, mutual_martix])
            matrix_gate_j = torch.einsum(
                'jk,ijk->ijk', [features_raw, mutual_martix])

            features_self_i = torch.einsum(
                'ik,ijk->ijk', [features_raw, matrix_gate_i]) + features_raw
            features_self_j = torch.einsum(
                'jk,ijk->ijk', [features_raw, matrix_gate_j]) + features_raw
            features_other_i = torch.einsum(
                'ik,ijk->ijk', [features_raw, matrix_gate_j]) + features_raw
            features_other_j = torch.einsum(
                'jk,ijk->ijk', [features_raw, matrix_gate_i]) + features_raw

            # 归一化 便于计算 余弦相似度
            # features_self_i NxNx2048 ij对比后增强的 特征i
            features_self_i = F.normalize(features_self_i, dim=2)
            features_self_j = F.normalize(features_self_j, dim=2)
            features_other_i = F.normalize(features_other_i, dim=2)
            features_other_j = F.normalize(features_other_j, dim=2)

            '''
                       exp(features_raw * features_self_i)
            Sum -log ---------------------------------------
                        Sum exp(features_raw * features_self_i) + Sum exp(features_raw * features_self_j)
                        + Sum exp(features_raw * features_other_i) + Sum exp(features_raw * features_other_j)
            新想法？设计方案
            1. 计算出所有的X-self 和 X-other
            2. 改两两对比为 1:N-1
            3. 计算新的对比损失函数
            4. 进行实验
            5. 分析其物理含义
            '''

            logits_mask = ((torch.eye(BS) - 1) * -1).cuda()

            raw_dot_self_i = torch.einsum(
                'ik,ijk->ij', [features_raw, features_self_i]) / self.temperature
            exp_raw_dot_self_i = torch.exp(raw_dot_self_i)

            raw_dot_self_j = torch.einsum(
                'ik,ijk->ij', [features_raw, features_other_j]) / self.temperature
            exp_raw_dot_self_j = torch.exp(raw_dot_self_j)

            raw_dot_other_i = torch.einsum(
                'ik,ijk->ij', [features_raw, features_other_i]) / self.temperature
            exp_raw_dot_other_i = torch.exp(raw_dot_other_i)

            raw_dot_other_j = torch.einsum(
                'ik,ijk->ij', [features_raw, features_other_j]) / self.temperature
            exp_raw_dot_other_j = torch.exp(raw_dot_other_j)

            sum_exp_raw_dot_self_i = torch.einsum('ij->i', exp_raw_dot_self_i)
            sum_exp_raw_dot_self_j = torch.einsum('ij->i', exp_raw_dot_self_j)
            sum_exp_raw_dot_other_i = torch.einsum(
                'ij->i', exp_raw_dot_other_i)
            sum_exp_raw_dot_other_j = torch.einsum(
                'ij->i', exp_raw_dot_other_j)

            # log_prob = torch.log(sum_exp_raw_dot_self_i) - torch.log(sum_exp_raw_dot_self_i + sum_exp_raw_dot_self_j + sum_exp_raw_dot_other_i + sum_exp_raw_dot_other_j)

            vs_all_exp_sum = torch.einsum('ij->i', exp_raw_dot_self_i + exp_raw_dot_self_j +
                                          exp_raw_dot_other_i + exp_raw_dot_other_j)
            vs_all_exp_sum = vs_all_exp_sum.view(-1, 1)

            log_prob = raw_dot_self_i - torch.log(vs_all_exp_sum)
            mean_log_prob_pos = log_prob.sum(1) / BS
            loss = - mean_log_prob_pos.mean()

            # mean_log_prob_pos = log_prob.sum(1)
            # loss = mean_log_prob_pos.mean()

            # projected_features = self.projector(features_raw)  # 2048 -> 512

            # 融合
            # for i, f in enumerate(features):
            #     g_Features[idxs[i].item()] = features[i].detach().cpu().numpy()
            #     g_Labels[idxs[i].item()] = targets[i].detach().cpu().numpy()

            # intra_pairs, inter_pairs, intra_labels, inter_labels = get_pairs(features_raw, labels)

            # features1 = torch.cat([features_raw[intra_pairs[:, 0]], features_raw[inter_pairs[:, 0]]], dim=0)
            # features2 = torch.cat([features_raw[intra_pairs[:, 1]], features_raw[inter_pairs[:, 1]]], dim=0)
            # features_raw_neg = features_raw[inter_pairs[:, 1]]

            # labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            # labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)

            # mutual_features = torch.cat([features_raw, features_raw_neg], dim=1)
            # map1_out = self.map1(mutual_features)
            # map2_out = self.drop(map1_out)
            # map2_out = self.map2(map2_out)

            # inter_pairs_feature = features[inter_pairs[:, 0]], features[inter_pairs[:, 1]]
            # intra_pairs_feature = features[intra_pairs[:, 0]], features[intra_pairs[:, 1]]

            # return raw_logits, _, features_raw, projected_features, intra_pairs_feature, inter_pairs_feature
            return raw_logits, _, features_raw, loss
        else:
            # BS = images.size(0)
            raw_logits, _, features_raw = self.pretrained_model(images)
            # return self.pretrained_model(images)
            return raw_logits

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
