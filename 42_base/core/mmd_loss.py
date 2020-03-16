import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .mmd import mix_rbf_mmd2

class MMDLoss(nn.Module):
    def __init__(self, base=1.0, sigma_list=[1, 2, 10]):
        super(MMDLoss, self).__init__()
        # sigma for MMD
        #         self.sigma_list = sigma_list
        self.base = base
        self.sigma_list = sigma_list
        self.sigma_list = [sigma / self.base for sigma in self.sigma_list]

    def forward(self, Target, Source):
        Target = Target.view(Target.size()[0], -1)
        Source = Source.view(Source.size()[0], -1)
        mmd2_D = mix_rbf_mmd2(Target, Source, self.sigma_list)
        mmd2_D = F.relu(mmd2_D)
        mmd2_D = torch.sqrt(mmd2_D)
        return mmd2_D


def loss_mmd_func(target_feature, source_feature):
    # We reserve the rights to provide the optimal sigma list
    criterion = MMDLoss(sigma_list=[1, 2, 10])
    loss = criterion(target_feature, source_feature)
    return loss