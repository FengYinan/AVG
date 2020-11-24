import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
#import tensorflow as tf
from torch.autograd import Variable
import math
from functools import partial

from retrying import retry

from utils import *

import C3D_model

from google.cloud import videointelligence
from google.cloud import storage

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

##################################################################################
# RestNet & Resnext
##################################################################################

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

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

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

##################################################################################
# P3D
##################################################################################

def conv_S(in_planes, out_planes, stride=1, padding=1):
    # as is descriped, conv S is 1x3x3
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=1,
                     padding=padding, bias=False)


def conv_T(in_planes, out_planes, stride=1, padding=1):
    # conv T is 3x1x1
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=1,
                     padding=padding, bias=False)


class Bottleneckp3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_s=0, depth_3d=47, ST_struc=('A', 'B', 'C')):
        super(Bottleneckp3d, self).__init__()
        self.downsample = downsample
        self.depth_3d = depth_3d
        self.ST_struc = ST_struc
        self.len_ST = len(self.ST_struc)

        stride_p = stride
        if not self.downsample == None:
            stride_p = (1, 2, 2)
        if n_s < self.depth_3d:
            if n_s == 0:
                stride_p = 1
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False, stride=stride_p)
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            if n_s == self.depth_3d:
                stride_p = 2
            else:
                stride_p = 1
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride_p)
            self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.id = n_s
        self.ST = list(self.ST_struc)[self.id % self.len_ST]
        if self.id < self.depth_3d:
            self.conv2 = conv_S(planes, planes, stride=1, padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(planes)
            #
            self.conv3 = conv_T(planes, planes, stride=1, padding=(1, 0, 0))
            self.bn3 = nn.BatchNorm3d(planes)
        else:
            self.conv_normal = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_normal = nn.BatchNorm2d(planes)

        if n_s < self.depth_3d:
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm3d(planes * 4)
        else:
            self.conv4 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def ST_A(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x

    def ST_B(self, x):
        tmp_x = self.conv2(x)
        tmp_x = self.bn2(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x + tmp_x

    def ST_C(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        tmp_x = self.conv3(x)
        tmp_x = self.bn3(tmp_x)
        tmp_x = self.relu(tmp_x)

        return x + tmp_x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        if self.id < self.depth_3d:  # C3D parts:

            if self.ST == 'A':
                out = self.ST_A(out)
            elif self.ST == 'B':
                out = self.ST_B(out)
            elif self.ST == 'C':
                out = self.ST_C(out)
        else:
            out = self.conv_normal(out)  # normal is res5 part, C2D all.
            out = self.bn_normal(out)
            out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class P3D(nn.Module):

    def __init__(self, block, layers, modality='RGB',
        shortcut_type='B', num_classes=400,dropout=0.5,ST_struc=('A','B','C')):
        self.inplanes = 64
        super(P3D, self).__init__()
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
        #                        padding=(3, 3, 3), bias=False)
        self.input_channel = 3 if modality=='RGB' else 2  # 2 is for flow
        self.ST_struc=ST_struc

        self.conv1_custom = nn.Conv3d(self.input_channel, 64, kernel_size=(1,7,7), stride=(1,2,2),
                                padding=(0,3,3), bias=False)

        self.depth_3d=sum(layers[:3])# C3D layers are only (res2,res3,res4),  res5 is C2D

        self.bn1 = nn.BatchNorm3d(64) # bn1 is followed by conv1
        self.cnt=0
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0,1,1))       # pooling layer for conv1.
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(2,1,1),padding=0,stride=(2,1,1))   # pooling layer for res2, 3, 4.

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=1)                              # pooling layer for res5.
        self.dropout=nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # some private attribute
        self.input_size=(self.input_channel,16,160,160)       # input of the network
        self.input_mean = [0.485, 0.456, 0.406] if modality=='RGB' else [0.5]
        self.input_std = [0.229, 0.224, 0.225] if modality=='RGB' else [np.mean([0.229, 0.224, 0.225])]


    @property
    def scale_size(self):
        return self.input_size[2] * 256 // 160   # asume that raw images are resized (340,256).

    @property
    def temporal_length(self):
        return self.input_size[1]

    @property
    def crop_size(self):
        return self.input_size[2]

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        stride_p=stride #especially for downsample branch.

        if self.cnt<self.depth_3d:
            if self.cnt==0:
                stride_p=1
            else:
                stride_p=(1,2,2)
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride_p, bias=False),
                        nn.BatchNorm3d(planes * block.expansion)
                    )

        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(planes * block.expansion)
                    )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
        self.cnt+=1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
            self.cnt+=1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.maxpool_2(self.layer1(x))  #  Part Res2
        x = self.maxpool_2(self.layer2(x))  #  Part Res3
        x = self.maxpool_2(self.layer3(x))  #  Part Res4

        sizes=x.size()
        x = x.view(-1,sizes[1],sizes[3],sizes[4])  #  Part Res5
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(-1,self.fc.in_features)
        x = self.fc(self.dropout(x))

        return x


def P3D199(pretrained=False,modality='RGB',**kwargs):
    """construct a P3D199 model based on a ResNet-152-3D model.
    """
    model = P3D(Bottleneckp3d, [3, 8, 36, 3], modality=modality,**kwargs)
    return model

##################################################################################
# Model
##################################################################################

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.in1 = nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.in2 = nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.in1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.in2(y)
        return x + y

class downlayer(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(downlayer, self).__init__()
        self.conv = nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=2, padding=1, bias=True)
        self.ins = nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True)
        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.ins(x)
        x = self.relu(x)
        return x

class uplayer(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(uplayer, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1, bias=True)
        self.ins = nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = F.interpolate(x, [x.size()[2] * 4, x.size()[3] * 4])
        x = self.conv(x)
        x = self.ins(x)
        x = self.relu(x)
        return x

class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=1, lable_num=2, n_res=6):
        super(Generator, self).__init__()
        self.embeding = nn.Embedding(lable_num, c_dim)

        self.conv1 = nn.Conv3d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=0, bias=True)
        self.in1 = nn.InstanceNorm3d(conv_dim, affine=True, track_running_stats=True)
        self.relu1 = nn.LeakyReLU(inplace=True)

        # Down-sampling layers.
        self.down, curr_dim = self.make_layer(downlayer, conv_dim, 4, "up")

        self.reslayer, curr_dim = self.make_layer(ResidualBlock, curr_dim, n_res, "equal")

        self.up, curr_dim = self.make_layer(uplayer, curr_dim, 4, "down")

        self.conv2 = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=0, bias=True)

        self.weight = nn.Parameter(torch.tensor(20., dtype=torch.float32))

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = self.embeding(c)
        c = torch.reshape(c, (-1,c.size()[-1],1,1,1))
        c = c.repeat(1, 1, x.size(2), x.size(3), x.size(4))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)

        x = self.down(x)

        x = x[:, :, 0, :]

        x = self.reslayer(x)

        x = self.up(x)

        x = F.pad(x, (3,3,3,3), mode='reflect')
        x = self.conv2(x)
        x = torch.tanh(x)

        x = torch.reshape(x, (x.size(0),x.size(1),1,x.size(2),x.size(3)) )* self.weight

        return x, c

    def make_layer(self, block, dim, layers, up):
        layer = []
        curr_dim = dim
        for i in range(layers):
            if up == "up":
                layer.append(block(curr_dim, curr_dim * 2))
                curr_dim = curr_dim * 2
            elif up == "down":
                layer.append(block(curr_dim, curr_dim // 2))
                curr_dim = curr_dim // 2
            else:
                layer.append(block(curr_dim, curr_dim))
        return nn.Sequential(*layer), curr_dim

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, c_dim=5):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(3 + c_dim, conv_dim, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu1 = nn.LeakyReLU(0.2,inplace=False)

        # Down-sampling layers.
        self.down, curr_dim = self.make_layer(downlayer, conv_dim, 2, "up")


        self.conv2 = nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.in1 = nn.InstanceNorm3d(curr_dim * 2, affine=True, track_running_stats=True)
        self.relu2 = nn.LeakyReLU(0.2,inplace=False)

        self.conv3 = nn.Conv3d(curr_dim * 2, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.down(x)
        x = self.conv2(x)
        x = self.in1(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x

    def make_layer(self, block, dim, layers, up):
        layer = []
        curr_dim = dim
        for i in range(layers):
            if up == "up":
                layer.append(block(curr_dim, curr_dim * 2))
                curr_dim = curr_dim * 2
            elif up == "down":
                layer.append(block(curr_dim, curr_dim // 2))
                curr_dim = curr_dim // 2
            else:
                layer.append(block(curr_dim, curr_dim))
        return nn.Sequential(*layer), curr_dim

def Classifier(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)#P3D199(pretrained=True,num_classes=600)#ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model

def Blackbox(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = C3D_model.C3D(num_classes=1, pretrained=False)#ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

class Googel():
    def __init__(self):
        self.video_client = videointelligence.VideoIntelligenceServiceClient()
        self.features = [videointelligence.enums.Feature.LABEL_DETECTION]

        self.bucket_id = 'avg_video'
        self.destination_blob_name = 'input.avi'
        self.storage_URI = 'gs://' + self.bucket_id + '/' + self.destination_blob_name


        self.storage_client = storage.Client()
        self.bucket = self.storage_client.get_bucket(self.bucket_id)
        self.blob = self.bucket.blob(self.destination_blob_name)


    def __call__(self, np_array, v_dir, target):
        lable = 'None'

        all_lable = ''

        path = image_to_video(np_array.cpu().detach().numpy().transpose((0, 2, 3, 4, 1)), v_dir)

        #with io.open(path, 'rb') as movie:
            #input_content = movie.read()
        self.blob.upload_from_filename(path)
        #self.blob.make_public()

        self.operation = self.video_client.annotate_video(self.storage_URI, features=self.features)#, input_content=input_content)
        print('Processing video for label annotations:')

        result = self.annotate_result()
        print('Finished processing.')

        if hasattr(result.annotation_results[0], error):
            print(result.annotation_results[0].error)

        segment_labels = result.annotation_results[0].segment_label_annotations
        for i, segment_label in enumerate(segment_labels):
            lable = segment_label.entity.description
            all_lable = all_lable + lable + '\t'

            if lable == target:

                for i, segment in enumerate(segment_label.segments):
                    confidence = segment.confidence

                    print(all_lable)
                    return lable, confidence

        print(all_lable)
        return lable, 0.

    def upload_blob(self, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""

        blob = self.bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

    @retry(wait_fixed=2000)
    def annotate_result(self):
        result = self.operation.result(timeout=1000)
        return  result


##################################################################################
# Loss function
##################################################################################

def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -torch.mean(real)
        fake_loss = torch.mean(fake)

    if loss_func == 'lsgan' :
        real_loss = torch.mean(torch.pow(real-1.0, 2))
        fake_loss = torch.mean(torch.pow(fake, 2))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = F.binary_cross_entropy_with_logits(real, torch.zeros_like(real)+1.)
        fake_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))

    if loss_func == 'hinge' :
        real_loss = torch.mean(F.leaky_relu_(1.0 - real))
        fake_loss = torch.mean(F.leaky_relu_(1.0 + fake))

    loss = real_loss + fake_loss

    return loss.to(DEVICE)

def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -torch.mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = torch.mean(torch.pow(fake-1.0, 2))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)+1.)

    if loss_func == 'hinge' :
        fake_loss = -torch.mean(fake)

    loss = fake_loss

    return loss.to(DEVICE)

def L1_loss(x, y):
    loss = torch.mean(torch.abs(x - y))

    return loss.to(DEVICE)

def L2_loss(x, y, c):
    loss = torch.clamp(torch.mean(torch.pow(x-y, 2)) - c, min=0.)

    return loss.to(DEVICE)

def adv_loss(logit, target_lable, c = 0, num_labels = 101, black = False):
    if black:
        loss = -torch.mean(logit)
        return loss.to(DEVICE)

    target_lable = label2onehot(target_lable, num_labels )
    target = torch.sum(torch.mul(target_lable, logit), 1).to(DEVICE)
    other, _ = torch.max((1.-target_lable) * logit - torch.mul(target_lable, 500000.), dim=1)
    other = other.to(DEVICE)
    loss = torch.mean(torch.clamp(other - target + c, 0.0))

    return loss.to(DEVICE)

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out.to(DEVICE)

def cross_entropy(output, lable):
    loss = torch.mean(-1 * lable * torch.log(output + 1e-6) - (1-lable) * torch.log(1-output + 1e-6))
    return  loss.to(DEVICE)

def target_lable_B(lable_real, lable_fake, target_lable, num_labels=101):
    target_lable = label2onehot(target_lable, num_labels)
    real = torch.sum(torch.mul(target_lable, lable_real), 1).to(DEVICE)
    fake = torch.sum(torch.mul(target_lable, lable_fake), 1).to(DEVICE)
    return  real, fake

