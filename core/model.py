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

    def forward(self, X, flag="train"):
        if flag == "train":
            X1, X2 = X
            batch = X1.size(0)

            # resnet_out, rpn_feature, feature = self.pretrained_model(x)
            resnet_out1, _, raw_features1 = self.pretrained_model(X1)
            resnet_out2, _, raw_features2 = self.pretrained_model(X2)
            return (resnet_out1, resnet_out2), _, (raw_features1, raw_features2)
        else:
            batch = X.size(0)
            return self.pretrained_model(X)
