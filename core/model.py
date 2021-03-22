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
        self.projector = projection_MLP(2048, 1024)

    def forward(self, X, flag="train"):
        if flag == "train":
            X1, X2 = X
            batch = X1.size(0)

            # resnet_out, rpn_feature, feature = self.pretrained_model(x)
            resnet_out1, _, raw_features1 = self.pretrained_model(X1)
            resnet_out2, _, raw_features2 = self.pretrained_model(X2)
            projected_features1 = self.projector(raw_features1)
            projected_features2 = self.projector(raw_features2)
            return (resnet_out1, resnet_out2), _, (projected_features1, projected_features2)
        else:
            batch = X.size(0)
            return self.pretrained_model(X)


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
