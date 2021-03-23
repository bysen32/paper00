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
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(2048, 200)
        )

        self.map1 = nn.Linear(2048 * 2, 512)
        self.map2 = nn.Linear(512, 2048)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, targets, flag="train"):
        if flag == "train":
            X1, X2 = X
            batch = X1.size(0)

            # resnet_out, rpn_feature, feature = self.pretrained_model(x)
            resnet_out1, _, raw_features1 = self.pretrained_model(X1)
            resnet_out2, _, raw_features2 = self.pretrained_model(X2)
            projected_features1 = self.projector(raw_features1)
            projected_features2 = self.projector(raw_features2)

            features = (raw_features1 + raw_features2) / 2
            intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(features, targets)
            features1 = torch.cat([features[intra_pairs[:, 0]], features[intra_pairs[:, 0]]], dim=0)
            features2 = torch.cat([features[intra_pairs[:, 1]], features[intra_pairs[:, 1]]], dim=0)

            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)

            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)

            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)
            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            features1_self = torch.mul(gate1, features1) + features1
            features1_other = torch.mul(gate2, features1) + features1
            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2

            logit1_self = self.fc(self.drop(features1_self))
            logit1_other = self.fc(self.drop(features1_other))
            logit2_self = self.fc(self.drop(features2_self))
            logit2_other = self.fc(self.drop(features2_other))
            pairs_logits = (logit1_self, logit1_other, logit2_self, logit2_other)
            pairs_labels = (labels1, labels2)

            return (resnet_out1, resnet_out2), _, (projected_features1, projected_features2), pairs_logits, pairs_labels
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
        distance_matrix = pdist(embeddings).detach().cpu().numpy()

        labels = labels.detach().cpu().numpy().reshape(-1, 1)
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
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
        inter_pairs = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])

        for i in range(embeddings.shape[0]):
            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]

            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]

            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]

            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]

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

