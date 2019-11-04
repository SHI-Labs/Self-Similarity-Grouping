from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import norm
import numpy as np

class WeightCE(nn.Module):
    def __init__(self, margin=0, num_instances=0, use_semi=True):
        super(WeightCE, self).__init__()
        self.margin = margin
        self.use_semi = use_semi
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, w):
        assert inputs.size(0) == w.size(0)
        loss = 0.
        for i in range(inputs.size(0)):
            loss += w[i] * self.cross_entropy(inputs[i].unsqueeze(0), targets[i].unsqueeze(0))
        loss /= inputs.size(0)
        return loss


