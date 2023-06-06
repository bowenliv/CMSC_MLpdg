import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter

class SCC_Conv2d(nn.Module):
    def __init__(self, num, in_channels, out_channels, kernel_size, stride=1,
                                 padding=0, dilation=1, groups=1, bias=True):
        super(SCC_Conv2d, self).__init__()
        self.num = num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = Parameter(torch.Tensor(out_channels * in_channels * kernel_size * kernel_size // groups, num))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs, inputs_se):
        batchsize, channel, height, width = inputs.shape
        weight = F.linear(inputs_se, self.weight)
        weight = weight.reshape(batchsize * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        inputs = inputs.reshape(1, batchsize * channel, height, width)
        outputs = F.conv2d(inputs, weight, None, self.stride, self.padding, self.dilation, groups=self.groups * batchsize)
        height, width = outputs.shape[2:]
        outputs = outputs.reshape(batchsize, self.out_channels, height, width)
        if self.bias is not None:
            outputs = outputs + self.bias.reshape(1, -1, 1, 1)
        return outputs

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_scc(num, in_planes, out_planes, stride=1):
    """3x3 dls convolution with padding"""
    return SCC_Conv2d(num, in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1_scc(num, in_planes, out_planes, stride=1):
    """1x1 dls convolution"""
    return SCC_Conv2d(num, in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, num=8):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1_scc(num, inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1_scc(num, planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.fc = nn.Linear(inplanes, num)

    def forward(self, x):
        identity = x

        x_se = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1, keepdim=False)
        x_se = F.sigmoid(self.fc(x_se))

        out = self.conv1(x, x_se)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, x_se)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=512, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and name == 'conv1':
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) and name != 'conv1':
                _, c, h, w = m.weight.shape
                nn.init.normal_(m.weight, 0, math.sqrt(1.0 / (c * h * w)))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, Dynamic_localshare_Conv2d) and name == 'conv1':
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)
            # elif isinstance(m, Dynamic_localshare_Conv2d) and name != 'conv1':
            #     c, h, w = m.in_channels, m.kernel_size[0], m.kernel_size[1]
            #     nn.init.normal_(m.weight, 0, math.sqrt(1.0 / (c * h * w)))
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)

            elif isinstance(m, SCC_Conv2d) and name == 'conv1':
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, SCC_Conv2d) and name != 'conv1':
                c, h, w = m.in_channels, m.kernel_size, m.kernel_size
                nn.init.normal_(m.weight, 0, math.sqrt(1.0 / (c * h * w)))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

if __name__ == '__main__':
    model = resnet50()
    flops, params = profile(model, input_size=(1,3,224,224), custom_ops={Soft_conditional_Conv2d:count_soft_conv2d})
    print('flops = {} params = {}'.format(clever_format(flops), clever_format(params)))

