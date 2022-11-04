from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from torchvision import models
from torch.hub import load_state_dict_from_url

# ResNet50 - V1


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html


class Bottleneck(nn.Module):
    expansion = 4  # this factor is the 3 dim expansion factor for each bottleneck block, for the next block, inplanes = expansion*planes

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes*(base_width/64.))*groups # ?? no grouping for normal resnet

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)  # modify the input directly
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Base(nn.Module):
    def __init__(self, layers, num_classes=1000, zero_init_residual=False,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(Base, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        # image input has 3 channels
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 64 is the base width, and layers[0] is the number of Bottleneck to build
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(
            Bottleneck, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            Bottleneck, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            Bottleneck, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*Bottleneck.expansion, num_classes)

        # initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # Zero-initialize the last BN in each residual branch,
                    nn.init.constant_m(m.bn3.weight, 0)
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                norm_layer(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                      dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes*block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1) # start flatting from dim = 1, dim = 0 for batch axis
        x = torch.nn.Flatten()(x) # flatten as layer
        x = self.fc(x)

        return x

def _base(layers, pretrained = False, progress= True, **kwargs):
    base = Base(layers, **kwargs)
    # print(base)
    # print(base.fc.weight)
    # print(base.input_shape)
    # print(base.output)
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',
                                              progress=progress)
        base.load_state_dict(state_dict)
    # print(model.fc.weight)
    return base

class CusNet(nn.Module):
    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(CusNet, self).__init__()
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        self.base = _base([3, 4, 6, 3], pretrained=pretrained, progress = True)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

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

    def reset_params(self):
        for m in self.modules():
            # base model already initialized in Base class
            # if isinstance(m, nn.Conv2d):
            #     init.kaiming_normal(m.weight, mode='fan_out')
            #     if m.bias is not None:
            #         init.constant(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     init.constant(m.weight, 1)
            #     init.constant(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     init.normal(m.weight, std=0.001)
            #     if m.bias is not None:
            #         init.constant(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std = 0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
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

if __name__=="__main__":
    base_model = _base([3, 4, 6, 3], pretrained= True, progress= True)

    # Testing:
    print("Testing:")
    input = torch.randn(5,3,224,224)
    model.eval()
    output = model(input)
    print(output)
    # print(model)
    