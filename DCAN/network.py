"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import torch
import torch.nn as nn
import pretrainedmodels
from torchvision import models

__all__ = ['SENet', 'se_resnet50', 'se_resnet101']

pretrained_settings = {
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        if self.training:
            src = self.fc0(x[:int(x.size(0) / 2), ])
            # src = self.fc1(x[:int(x.size(0) / 2), ])
            trg = self.fc1(x[int(x.size(0) / 2):, ])
            x = torch.cat((src, trg), 0)
        else:
            x = self.fc1(x)

        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1):
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64, track_running_stats=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64, track_running_stats=True)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes, track_running_stats=True)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes, track_running_stats=True)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True` is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None

        self.feature = nn.Sequential(self.layer0, self.layer1, self.layer2, self.layer3, self.layer4)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion, track_running_stats=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.feature(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, settings):
    # assert num_classes == settings['num_classes'], \
    #     'num_classes should be {}, but is {}'.format(
    #         settings['num_classes'], num_classes)

    # model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def se_resnet50(pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, settings)

    pretrained_dict = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet').state_dict()

    model_dict = model.state_dict()
    # filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items()
    #                    if k and k != 'last_linear.weight' and k != 'last_linear.bias' in model_dict}
    # overwrite entries in the existing state dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict}
    layer = [3, 4, 6, 3]
    model_dict.update(pretrained_dict)
    for i in range(1, 5):
        for j in range(layer[i - 1]):
            a1 = 'layer{}.{}.se_module.fc0.weight'.format(i, j)
            a2 = 'layer{}.{}.se_module.fc0.bias'.format(i, j)
            b1 = 'layer{}.{}.se_module.fc1.weight'.format(i, j)
            b2 = 'layer{}.{}.se_module.fc1.bias'.format(i, j)
            model_dict[a1] = pretrained_dict[b1]
            model_dict[a2] = pretrained_dict[b2]
    # load the new state dict
    model.load_state_dict(model_dict)
    return model


def se_resnet101(pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, settings)

    pretrained_dict = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet').state_dict()

    model_dict = model.state_dict()
    # filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items()
    #                    if k and k != 'last_linear.weight' and k != 'last_linear.bias' in model_dict}
    # overwrite entries in the existing state dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict}
    layer = [3, 4, 23, 3]
    model_dict.update(pretrained_dict)
    for i in range(1, 5):
        for j in range(layer[i - 1]):
            a1 = 'layer{}.{}.se_module.fc0.weight'.format(i, j)
            a2 = 'layer{}.{}.se_module.fc0.bias'.format(i, j)
            b1 = 'layer{}.{}.se_module.fc1.weight'.format(i, j)
            b2 = 'layer{}.{}.se_module.fc1.bias'.format(i, j)
            model_dict[a1] = pretrained_dict[b1]
            model_dict[a2] = pretrained_dict[b2]
    # load the new state dict
    model.load_state_dict(model_dict)
    return model


def se_resnet152(pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16, dropout_p=None, inplanes=64,
                  input_3x3=False, downsample_kernel_size=1, downsample_padding=0)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, settings)

    pretrained_dict = pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained='imagenet').state_dict()

    model_dict = model.state_dict()
    # filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items()
    #                    if k and k != 'last_linear.weight' and k != 'last_linear.bias' in model_dict}
    # overwrite entries in the existing state dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict}
    layer = [3, 8, 36, 3]
    model_dict.update(pretrained_dict)
    for i in range(1, 5):
        for j in range(layer[i - 1]):
            a1 = 'layer{}.{}.se_module.fc0.weight'.format(i, j)
            a2 = 'layer{}.{}.se_module.fc0.bias'.format(i, j)
            b1 = 'layer{}.{}.se_module.fc1.weight'.format(i, j)
            b2 = 'layer{}.{}.se_module.fc1.bias'.format(i, j)
            model_dict[a1] = pretrained_dict[b1]
            model_dict[a2] = pretrained_dict[b2]
    # load the new state dict
    model.load_state_dict(model_dict)
    new_modle_dic = model.state_dict()
    return model

class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

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
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

