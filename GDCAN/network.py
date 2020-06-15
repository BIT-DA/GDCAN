"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import models

__all__ = ['DCCANet', 'dcca_resnet50', 'dcca_resnet101', 'dcca_resnet152']

pretrained_settings = {
    'dcca_resnet50': {
        'imagenet': {
            'RESTORE_FROM': 'pretrained_models/dcca_resnet50_pretrained_imagenet.pth'
        }
    },
    'dcca_resnet101': {
        'imagenet': {
            'RESTORE_FROM': 'pretrained_models/dcca_resnet101_pretrained_imagenet.pth'
        }
    },
    'dcca_resnet152': {
        'imagenet': {
            'RESTORE_FROM': 'pretrained_models/dcca_resnet152_pretrained_imagenet.pth'
        }
    },
}


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


class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        self.restored = False
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


class DCCAModule(nn.Module):

    def __init__(self, channels, reduction, gate_eta):
        super(DCCAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()
        self.gate = False
        self.gate_eta = gate_eta

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)

        if self.training:
            src = x[:int(x.size(0) / 2), ]
            trg = x[int(x.size(0) / 2):, ]

            src_dev = src.mean() / (src.var() + 1e-5).sqrt()
            trg_dev = trg.mean() / (trg.var() + 1e-5).sqrt()
            dis = torch.abs(src_dev - trg_dev)
            prob = torch.tanh(dis / trg_dev)

            if prob < self.gate_eta:
                self.gate = True
                trg = self.fc0(trg)
            else:
                self.gate = False
                trg = self.fc1(trg)

            src = self.fc0(src)

            x = torch.cat((src, trg), 0)

        else:
            # if self.output:
            # print(self.gate)
            if self.gate:
                x = self.fc0(x)
            else:
                x = self.fc1(x)

        x = self.relu(x)
        x = self.fc2(x)

        x = self.sigmoid(x)
        return module_input * x


class DCCAResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, gate_eta, stride=1,
                 downsample=None):
        super(DCCAResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = DCCAModule(planes * 4, reduction=reduction, gate_eta=gate_eta)
        self.downsample = downsample
        self.stride = stride


class DCCANet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, gate_eta=0.2):
        super(DCCANet, self).__init__()
        self.inplanes = inplanes
        self.gate_eat = gate_eta
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
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
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
        # bottleneck 2048 * 256
        # self.bottleneck = nn.Linear(512 * block.expansion, 256)

        # self.last_linear = nn.Linear(256, num_classes)

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
        layers.append(block(self.inplanes, planes, groups, reduction, self.gate_eat, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction, self.gate_eat))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        # RevGrad
        # feature = self.bottleneck(x)
        # feature = feature.view(-1, 256)

        # x = self.last_linear(feature)
        # x = features
        return x

    def forward(self, x):
        # x = self.features(x)
        x = self.feature(x)
        x = self.logits(x)
        return x



def dcca_resnet50(gate_eta=0.2, pretrained='imagenet'):
    model = DCCANet(DCCAResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                    dropout_p=None, inplanes=64, input_3x3=False,
                    downsample_kernel_size=1, downsample_padding=0, gate_eta=gate_eta)
    if pretrained is not None:
        settings = pretrained_settings['dcca_resnet50'][pretrained]
        model.load_state_dict(torch.load(settings['RESTORE_FROM']))

    return model


def dcca_resnet101(gate_eta=0.2, pretrained='imagenet'):
    model = DCCANet(DCCAResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                    dropout_p=None, inplanes=64, input_3x3=False,
                    downsample_kernel_size=1, downsample_padding=0, gate_eta=gate_eta)
    if pretrained is not None:
        settings = pretrained_settings['dcca_resnet101'][pretrained]
        model.load_state_dict(torch.load(settings['RESTORE_FROM']))
    return model


def dcca_resnet152(gate_eta=0.2, pretrained='imagenet'):
    model = DCCANet(DCCAResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                    dropout_p=None, inplanes=64, input_3x3=False,
                    downsample_kernel_size=1, downsample_padding=0, gate_eta=gate_eta)
    if pretrained is not None:
        settings = pretrained_settings['dcca_resnet152'][pretrained]
        model.load_state_dict(torch.load(settings['RESTORE_FROM']))

    return model
