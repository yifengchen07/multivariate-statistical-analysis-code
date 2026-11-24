"""ImageNet-style ResNet-18 with a configurable base channel count."""
import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None if stride == 1 and inplanes == planes else nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class WidthResNet18(nn.Module):
    def __init__(self, width=64, classes=4):
        super().__init__()
        self.inplanes = width
        self.conv1 = nn.Conv2d(3, width, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._layer(width, 1)
        self.layer2 = self._layer(width * 2, 2)
        self.layer3 = self._layer(width * 4, 2)
        self.layer4 = self._layer(width * 8, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8, classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _layer(self, planes, stride):
        blocks = [BasicBlock(self.inplanes, planes, stride), BasicBlock(planes, planes)]
        self.inplanes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def build_model(config, width):
    assert config['architecture'] == 'WidthResNet18' and config['image_size'] == 224
    return WidthResNet18(width, classes=len(config['class_names']))
