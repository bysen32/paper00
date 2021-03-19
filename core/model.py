from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet
import numpy as np

class MyNet(nn.Module):
    def __init__(self):
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)
    
    def forward(self, x):
        batch = x.size(0)

        resnet_out, rpn_feature, feature = self.pretrained_model(x)
