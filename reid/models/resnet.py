from __future__ import absolute_import

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
import torchvision
from reid.models.dce import ClusterAssignment
# from torch_deform_conv.layers import ConvOffset2D
from reid.utils.serialization import load_checkpoint, save_checkpoint

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
        # 18: base.resnet18,
        # 34: base.resnet34,
        # 50: base.resnet50,
        # 101: base.resnet101,
        # 152: base.resnet152,
    }

    def __init__(self, depth, checkpoint=None, pretrained=True, num_features=2048, 
                    dropout=0.1, num_classes=0, num_split=1, mode='Dissimilarity', cluster=False):
        super(ResNet, self).__init__()

        self.depth = depth
        self.checkpoint = checkpoint
        self.pretrained = pretrained
        self.num_features = num_features
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_split = num_split
        self.cluster = cluster

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)
        out_planes = self.base.fc.in_features

        # resume from pre-iteration training 
        if self.checkpoint:
            state_dict = load_checkpoint(checkpoint)
            self.load_state_dict(state_dict['state_dict'], strict=False)

        # In deep person has_embedding is always False. So self.num_features==out_planes
        # for x1 bn
        #self.x1_bn = nn.BatchNorm1d(2048)
        #init.constant(self.x1_bn.weight,1)
        #init.constant(self.x1_bn.bias,0)
        # for x2 embedding
        if self.num_features > 0:
            self.feat = nn.Linear(out_planes, self.num_features, bias=False)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.relu = nn.ReLU(inplace=True)
            init.normal_(self.feat.weight, std=0.001)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)

        #x2 classifier
        if self.num_classes > 0:
            self.classifier_x2 = nn.Linear(self.num_features, self.num_classes)
            init.normal_(self.classifier_x2.weight, std=0.001)
            init.constant_(self.classifier_x2.bias, 0)
        if self.cluster:
            self.assignment = ClusterAssignment(cluster_number=32, embedding_dimension=2048)
            # if self.num_split <= 1:
            #     self.assignment = ClusterAssignment(cluster_number=32, embedding_dimension=2048)
            # else:
            #     self.assignment = nn.ModuleList([ClusterAssignment(cluster_number=32, embedding_dimension=2048) for i in range(self.num_split+1)])

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, for_eval=False):
        for name, module in self.base._modules.items():
            if name == 'layer4':
                f_m = x
            if name == 'avgpool':
                break
            x = module(x)
        if self.num_split > 1:
            h = x.size(2)
            x1 = []
            xx = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
            x1.append(xx)
            x1_split = [x[:, :, h // self.num_split * s: h // self.num_split * (s+1), :] for s in range(self.num_split)]
            # x1_split_2 = [x[:, :, h // (2*self.num_split) * s: h // (2*self.num_split) * (s+1), :] for s in range(self.num_split*2)]
            # h1 = x1_split[0].size(2)
            # x1_split_1 = [x1_split[0][:, :, h1 // self.num_split * s: h1 // self.num_split * (s+1), :] for s in range(self.num_split)]
            # x1_split_2 = [x1_split[1][:, :, h1 // self.num_split * s: h1 // self.num_split * (s+1), :] for s in range(self.num_split)]
            # x1.append(torch.cat([F.avg_pool2d(xx, xx.size()[2:]).view(xx.size(0), -1) for xx in x1_split], dim=1))
            # x1.append(torch.cat([F.avg_pool2d(xx, xx.size()[2:]).view(xx.size(0), -1) for xx in x1_split_2], dim=1))
            # x1.append(torch.cat([F.avg_pool2d(xx, xx.size()[2:]).view(xx.size(0), -1) for xx in x1_split_2], dim=1))
            for xx in x1_split:
                xx = F.avg_pool2d(xx, xx.size()[2:])
                x1.append(xx.view(xx.size(0), -1))          
        else:
            x1 = F.avg_pool2d(x, x.size()[2:])
            x1 = x1.view(x1.size(0), -1)
        if self.num_features > 0:
            x2 = F.avg_pool2d(x, x.size()[2:])
            x2 = x2.view(x2.size(0), -1)
            x2 = self.feat(x2)
            x2 = self.feat_bn(x2)
            x2 = self.relu(x2)
        if self.num_classes > 0:
            x2 = self.drop(x2)
            x2 = self.classifier_x2(x2)

        if for_eval and isinstance(x1, list):
            x1 = torch.cat(x1, dim=1)
            return x1, x2
        else:
            if self.cluster:
                if isinstance(x1, list):
                    # x3 = [self.assignment[i](x) for i, x in enumerate(x1)]
                    x3 = self.assignment(torch.cat(x1, dim=1))
                else:
                    x3 = self.assignment(x1)
                return x1, x2, x3
            else:
                return x1, x2

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
