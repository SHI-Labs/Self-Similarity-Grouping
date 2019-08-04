from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import norm

import numpy as np

class TripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=0, use_semi=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.use_semi = use_semi
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        self.K = num_instances

    def forward(self, inputs, targets, epoch, w=None):
        # if w is not None:
        #     inputs = inputs * w.unsqueeze(1)
        n = inputs.size(0)
        P = n // self.K
        t0 = 20.0
        t1 = 40.0
      
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        if False: ######## curriculum sampling
            mean = max(124-6.0*epoch, 0.0)
            #std = 12*0.001**((epoch-t0)/t0) if epoch >= t0 else 12
            std = 15*0.001**(max((epoch-t0)/(t1-t0), 0.0))
            neg_probs = norm(mean, std).pdf(np.linspace(0,123,124))
            neg_probs = torch.from_numpy(neg_probs).clamp(min=3e-5, max=20.0)
            for i in range(P):
                for j in range(self.K):
                    neg_examples = dist[i*self.K+j][mask[i*self.K+j] == 0]
                    #sort_neg_examples = torch.topk(neg_examples, k=80, largest=False)[0]
                    sort_neg_examples = torch.sort(neg_examples)[0]
                    for pair in range(j+1,self.K):
                        dist_ap.append(dist[i*self.K+j][i*self.K+pair])
                        choosen_neg = sort_neg_examples[torch.multinomial(neg_probs,1).cuda()]
                        dist_an.append(choosen_neg)
        elif self.use_semi:  ######## semi OHEM
            for i in range(P):
                for j in range(self.K):
                    neg_examples = dist[i*self.K+j][mask[i*self.K+j] == 0]
                    for pair in range(j+1,self.K):
                        ap = dist[i*self.K+j][i*self.K+pair]
                        dist_ap.append(ap.view(1))
                        dist_an.append(neg_examples.min().view(1))
        else:  ##### OHEM
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().view(1))
                dist_an.append(dist[i][mask[i] == 0].min().view(1))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y) 
        if w is not None:
            loss = 0.
            for i in range(dist_an.size(0)):
                loss += self.ranking_loss(dist_an[i].unsqueeze(0), dist_ap, y)
            loss /= dist_an.size(0)
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, epoch):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1*((1-pt)**self.gamma)*logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

'''
class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
'''
