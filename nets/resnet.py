'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MySequential(nn.Sequential):
    def forward(self, x, adv):
        for module in self._modules.values():
            x = module(x, adv=adv)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(BasicBlock, self).__init__()
        self.bn_adv_momentum = bn_adv_momentum
        self.bn_adv_flag = bn_adv_flag
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn2_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)

        self.shortcut = nn.Sequential()
        self.shortcut_bn = None
        self.shortcut_bn_adv = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )
            self.shortcut_bn = nn.BatchNorm2d(self.expansion * planes)
            if self.bn_adv_flag:
                self.shortcut_bn_adv = nn.BatchNorm2d(self.expansion * planes, momentum=self.bn_adv_momentum)

    def forward(self, x, adv=False):
        if adv and self.bn_adv_flag:
            out = F.relu(self.bn1_adv(self.conv1(x)))
            out = self.conv2(out)
            out = self.bn2_adv(out)
            if self.shortcut_bn_adv:
                out += self.shortcut_bn_adv(self.shortcut(x))
            else:
                out += self.shortcut(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.conv2(out)
            out = self.bn2(out)
            if self.shortcut_bn:
                out += self.shortcut_bn(self.shortcut(x))
            else:
                out += self.shortcut(x)

        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(Bottleneck, self).__init__()
        self.bn_adv_momentum = bn_adv_momentum
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn_adv_flag = bn_adv_flag

        self.bn1 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn2_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        if self.bn_adv_flag:
            self.bn3_adv = nn.BatchNorm2d(self.expansion * planes, momentum=self.bn_adv_momentum)

        self.shortcut = nn.Sequential()
        self.shortcut_bn = None
        self.shortcut_bn_adv = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )
            self.shortcut_bn = nn.BatchNorm2d(self.expansion * planes)
            if self.bn_adv_flag:
                self.shortcut_bn_adv = nn.BatchNorm2d(self.expansion * planes, momentum=self.bn_adv_momentum)

    def forward(self, x, adv=False):

        if adv and self.bn_adv_flag:

            out = F.relu(self.bn1_adv(self.conv1(x)))
            out = F.relu(self.bn2_adv(self.conv2(out)))
            out = self.bn3_adv(self.conv3(out))
            if self.shortcut_bn_adv:
                out += self.shortcut_bn_adv(self.shortcut(x))
            else:
                out += self.shortcut(x)
        else:

            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.shortcut_bn:
                out += self.shortcut_bn(self.shortcut(x))
            else:
                out += self.shortcut(x)

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, proj_size=128, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.bn_adv_momentum = bn_adv_momentum
        self.bn_adv_flag = bn_adv_flag
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(64, momentum = self.bn_adv_momentum)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, bn_adv_flag = self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum)

        self.pool = nn.AdaptiveAvgPool2d(1)
        #Non-linear
        self.linear = nn.Sequential(nn.Linear(512*block.expansion, 512*block.expansion*2, bias=False),
                                         nn.BatchNorm1d(int(512*block.expansion*2)),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(int(512*block.expansion*2), proj_size))

    def _make_layer(self, block, planes, num_blocks, stride, bn_adv_flag=False, bn_adv_momentum=0.01):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,  bn_adv_flag=bn_adv_flag, bn_adv_momentum = bn_adv_momentum))
            self.in_planes = planes * block.expansion
        return MySequential(*layers)

    def forward(self, x, adv = False, lin=0, lout=5):
        if adv and self.bn_adv_flag:
            out = F.relu(self.bn1_adv(self.conv1(x)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out, adv=adv)
        out = self.layer2(out, adv=adv)
        out = self.layer3(out, adv=adv)
        out = self.layer4(out, adv=adv)
        
        out = self.pool(out)
        feat = out.view(out.size(0), -1)
        
        out = self.linear(feat)
        out = F.normalize(out, p=2, dim=1)

        return out


def ResNet18(proj_size=10, bn_adv_flag=False, bn_adv_momentum=0.01):
    return ResNet(BasicBlock, [2,2,2,2], proj_size, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)

def ResNet34(low_dim=128, bn_adv_flag=False, bn_adv_momentum=0.01):
    return ResNet(BasicBlock, [3,4,6,3], low_dim, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)

def ResNet50(proj_size=10, bn_adv_flag=False, bn_adv_momentum=0.01):
    return ResNet(Bottleneck, [3,4,6,3], proj_size, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum)


