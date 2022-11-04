from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from torchvision import models


__all__ = ['Еfficientnet', 'efficientnet', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
            'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


class Еfficientnet(nn.Module):
    __factory = {
        "": models.efficientnet,
        "_b0": models.efficientnet_b0,
        "_b1": models.efficientnet_b1,
        "_b2": models.efficientnet_b2,
        "_b3": models.efficientnet_b3,
        "_b4": models.efficientnet_b4,
        "_b5": models.efficientnet_b5,
        "_b6": models.efficientnet_b6,
        "_b7": models.efficientnet_b7,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(Еfficientnet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) Еfficientnet
        if depth not in Еfficientnet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = Еfficientnet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.classifier[1].in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def efficientnet(**kwargs):
    return Еfficientnet("", **kwargs)


def efficientnet_b0(**kwargs):
    return Еfficientnet("_b0", **kwargs)


def efficientnet_b1(**kwargs):
    return Еfficientnet("_b1", **kwargs)


def efficientnet_b2(**kwargs):
    return Еfficientnet("_b2", **kwargs)


def efficientnet_b3(**kwargs):
    return Еfficientnet("_b3", **kwargs)

def efficientnet_b4(**kwargs):
    return Еfficientnet("_b4", **kwargs)


def efficientnet_b5(**kwargs):
    return Еfficientnet("_b5", **kwargs)


def efficientnet_b6(**kwargs):
    return Еfficientnet("_b6", **kwargs)

def efficientnet_b7(**kwargs):
    return Еfficientnet("_b7", **kwargs)