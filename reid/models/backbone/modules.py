# -*- coding: utf-8 -*-
# Time    : 2020/1/31 14:37
# Author  : Yichen Lu

import paddle
from copy import deepcopy
import paddle.nn as nn
from paddle.nn import functional as F


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, x, ret_attention=False):
        b, c, *dims = x.shape
        y = self.avg_pool(x).reshape([b, c])
        y = self.fc(y).reshape([b, c, 1, 1])
        if ret_attention:
            return x * y.expand_as(x), y.reshape([b, -1])
        else:
            return x * y.expand_as(x)


class SEBottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SeparateBN(nn.Layer):
    def __init__(self, bn, num_domains=2):
        super(SeparateBN, self).__init__()
        self.num_domains = num_domains

        assert isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2D, nn.BatchNorm3d, nn.SyncBatchNorm)), \
            "bn should be an instance of nn._BatchNorm."
        self.bns = nn.ModuleDict({str(i): deepcopy(bn) for i in range(num_domains)})

    def forward(self, input, domain_indices):
        if isinstance(domain_indices, paddle.Tensor):
            domains = domain_indices.unique()
        elif isinstance(domain_indices, int):
            batch_size = input.shape[0]
            domain_indices = paddle.ones(batch_size, dtype=paddle.int32) * domain_indices
            domains = domain_indices.unique()
        else:
            raise RuntimeError("Invalid domain_indices.")
        batches = []
        for domain in domains:
            indices = (domain_indices == domain).nonzero()
            from_same_domain = input[indices.squeeze()]
            batches.append(self.bns[str(domain.item())](from_same_domain))
        return paddle.concat(batches, axis=0)

    def clone(self, src_key, new_key):
        assert src_key in self.bns, "Source key not registered in ModuleDict."
        self.bns[new_key] = deepcopy(self.bns[src_key])


class Sequential(nn.Sequential):

    def __init__(self, *args):
        from reid.models.backbone.separable_resnet import Bottleneck
        self._separable_classes = (SeparateBN, Sequential, Bottleneck)
        super(Sequential, self).__init__(*args)

    def forward(self, input, domain_indices=None):
        for module in self:
            if isinstance(module, self._separable_classes):
                input = module(input, domain_indices)
            else:
                input = module(input)
        return input
