# -*- coding: utf-8 -*-
# Time    : 2020/1/17 22:24
# Author  : Yichen Lu

from os import path as osp
import paddle
from paddle import ParamAttr
from paddle import nn
from paddle.nn import functional as F
from reid.utils.utils import DivisibleDict
from reid.utils.kaimingnormal import kaiming_normal_
import sys


model_path = {
    'resnet34': './backbone_weights/resnet34-333f7ec4.pdparams',
    'resnet50': './backbone_weights/resnet50-19c8e357.pdparams',
}


class ResNet(nn.Layer):

    def __init__(self, depth=50, pretrained=True, last_stride=2, last_pooling='avg', embedding=256):
        super(ResNet, self).__init__()

        assert depth in (50, 34), "Invalid depth for ResNet, expect depth in (34, 50)."
        self.depth = depth
        self.pretrained = pretrained
        self.last_pooling = last_pooling
        self.last_stride = last_stride
        self.embedding = embedding
        self.forward_hooks, self.backward_hooks = [], []
        self.forward_records, self.backward_records = {}, {}

        # Construct base (pretrained) resnet
        block = Bottleneck if depth >= 50 else BasicBlock
        self.base = Base(block, layers=[3, 4, 6, 3], last_stride=self.last_stride)

        if pretrained:
            ckpt = model_path[f'resnet{depth}']
            assert osp.isfile(
                osp.expanduser(ckpt)), "paddle pretrained model doesn't exists."
            state_dict = paddle.load(osp.expanduser(ckpt))
            self.base.set_state_dict(state_dict)

        # 单独设置学习率 为初始的10倍
        self.bn = nn.BatchNorm1D(num_features=self.embedding,
                                 weight_attr=ParamAttr(learning_rate=10.0, initializer=nn.initializer.Constant(value=1.0)),
                                 bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.0), trainable=False))

        print(self)

    def forward(self, x, out=None, *kwargs):
        x1, x2, x3, feature_map = self.base(x)

        B, C, H, W = feature_map.shape

        if self.last_pooling == "max":
            feature = F.adaptive_max_pool2d(feature_map, 1).reshape([B, -1])
        else:
            feature = F.adaptive_avg_pool2d(feature_map, 1).reshape([B, -1])

        embedded_features = self.bn(feature)

        if out is not None:
            return DivisibleDict({'embedding': embedded_features.detach().cuda(out),
                                  'global': feature.detach().cuda(out),
                                  'maps': [x1.detach().cuda(out), x2.detach().cuda(out), x3.detach().cuda(out),
                                           feature_map.detach().cuda(out)],
                                  })

        return DivisibleDict({'embedding': embedded_features,
                              'global': feature,
                              'maps': [x1, x2, x3, feature_map],
                              })

    def create_forward_hook(self, name):
        def forward_hook(module, input, output):
            self.forward_records[name] = output
            return None

        return forward_hook

    def create_backward_hook(self, name):
        def backward_hook(module, grad_input, grad_output):
            self.backward_records[name] = grad_output[0]
            return None

        return backward_hook

    def remove_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()

        for hook in self.backward_hooks:
            hook.remove()
        print("Removed all hooks.")

    def __repr__(self):
        return f"Build ResNet{self.depth}(pretrained={self.pretrained}, last_stride={self.last_stride}, " \
               f"last_pooling={self.last_pooling}, embedding={self.embedding}) \n" \
               f"Modules: {self._sub_layers} \n"


class Base(nn.Layer):

    def __init__(self, block, layers, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, last_stride=2):
        super(Base, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                m.weight.set_value(paddle.ones_like(m.weight))
                m.bias.set_value(paddle.zeros_like(m.bias))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias_attr=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
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


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


if __name__ == '__main__':
    model = Base(Bottleneck, [3, 4, 6, 3])
    pass
