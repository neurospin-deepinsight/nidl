import torch
import torch.nn as nn
from typing import Tuple, Union, Type

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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

class ResNet(nn.Module):
    """ 3D ResNet architecture adapted from <https://arxiv.org/pdf/1512.03385.pdf>

    Parameters
    ----------
    block: Union[Type[BasicBlock], Type[Bottleneck]]
        Which convolution block to apply (4 in total). 
        This should be a class type and not its instance.
    
    layers: (int, int, int, int)
        How many layers in each conv block (4 in total).

    in_channels: int, default=1
        How many input channels has the input.

    zero_init_residual: bool, default=False
        Zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros, and each residual block 
        behaves like an identity. This improves the model by 0.2~0.3% 
        according to https://arxiv.org/abs/1706.02677
    
    groups: int, default=1
        How many groups to divide the input channels into in grouped 3x3 convolution
        of Bottleneck block (e.g. in Resnet50).

    width_per_group: int, default=64
        Number of channels per group in the grouped 3x3 convolution of 
        Bottleneck block (e.g. in ResNet50). Effective width = groups * width_per_group

    replace_stride_with_dilation: None or Tuple[bool, bool, bool], default=None
        By default, ResNet reduces spatial resolution of input feature maps using 
        stride 2 conv in layers 2, 3, 4. This replaces some layers with stride=2 by 
        dilation (atrous) conv, preserving spatial resolution. It is useful for dense
        tasks such as segmentation. 

    norm_layer: None or Type[nn.Module], default=None
        Which normalization to apply after each layer. If None, nn.BatchNorm3d is applied.

    initial_kernel_size: int, default=7
        Kernel size in the first conv layer.
    
    n_embedding: int, default=512
        Output dimension. 
    """

    def __init__(self, 
                 block: Union[Type[BasicBlock], Type[Bottleneck]], 
                 layers: Tuple[int, int, int, int], 
                 in_channels: int=1, 
                 zero_init_residual: bool=False,
                 groups: int=1, 
                 width_per_group: int=64, 
                 replace_stride_with_dilation: Union[None, Tuple[bool, bool, bool]]=None,
                 norm_layer: Union[None, Type[nn.Module]]=None, 
                 initial_kernel_size: int=7, 
                 n_embedding: int=512):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.name = "resnet"
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
        initial_stride = 2 if initial_kernel_size==7 else 1
        padding = (initial_kernel_size-initial_stride+1)//2
        self.conv1 = nn.Conv3d(in_channels, self.inplanes,
                               kernel_size=initial_kernel_size,
                               stride=initial_stride,
                               padding=padding, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        channels = [64, 128, 256, 512]

        self.layer1 = self._make_layer(block, channels[0], layers[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.embedding = nn.Linear(channels[3], n_embedding)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.embedding(x)
        return features


class ResnetTruncated(ResNet):

    def __init__(self, *args, depth: int, **kwargs):
        super().__init__(*args, **kwargs)
        if depth < 0 or depth > 4:
            raise ValueError("'depth' must be between 0 and 4 (included)")
        self.depth = depth
        self.first_conv = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
        )
        self.layers = nn.Sequential(
            self.layer1,  # Layer 1
            self.layer2,  # Layer 2
            self.layer3,  # Layer 3
            self.layer4   # Layer 4 (deepest before FC)
        )[:depth]  # Keep only the first 'depth' layers
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.layers(x)
        return x


def _resnet(arch: str, 
            block: nn.Module, 
            layers: Tuple[int, int, int, int], 
            **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def _resnet_trunc(arch: str, 
                  block: nn.Module, 
                  layers: Tuple[int, int, int, int], 
                  depth: int, 
                  **kwargs):
    model = ResnetTruncated(block, layers, depth=depth, **kwargs)
    return model


def resnet18(**kwargs):
    """3D ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`
    """
    return _resnet('resnet18', BasicBlock, (2, 2, 2, 2),  **kwargs)

def resnet50(**kwargs):
    """3D ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet50', Bottleneck, (3, 4, 6, 3),  **kwargs)


def resnet18_trunc(depth:int, **kwargs):
    """Truncated ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet_trunc('resnet18', BasicBlock, (2, 2, 2, 2),  depth, **kwargs)

def resnet50_trunc(depth: int, **kwargs):
    """Truncated ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet_trunc('resnet50', Bottleneck, (3, 4, 6, 3),  depth, **kwargs)

