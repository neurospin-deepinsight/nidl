import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from typing import Tuple

__all__ = ['DenseNet', '_densenet', 'densenet121']


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),

        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                memory_efficient=memory_efficient,
                )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""3D Densenet-BC model class, adapted from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Parameters
    ----------
    growth_rate: int, default=32
        How many filters to add at each layer (`k` in paper).
    
    block_config: (int, int, int, int), default=(3, 12, 24, 16)
        How many layers in each pooling block (4 blocks in total).
    
    num_init_features: int, default=64
        Number of filters to learn in the first convolution layer.
    
    bn_size: int, default=4 
        Multiplicative factor for number of bottleneck layers
        (i.e. bn_size * k features in the bottleneck layer)
    
    in_channels: int, default=1
        How many input channels has the input.
    
    n_embedding: int, default=512 
        Number of output dimensions
    
    memory_efficient: bool, default=False 
        If True, uses checkpointing. Much more memory efficient, 
        but slower. See <https://arxiv.org/pdf/1707.06990.pdf>
    """

    def __init__(self, 
                 growth_rate: int=32, 
                 block_config: Tuple[int, int, int, int]=(3, 12, 24, 16),
                 num_init_features: int=64, 
                 bn_size: int=4, 
                 in_channels: int=1, 
                 n_embedding: int=512,
                 memory_efficient: bool=False):
        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.embedding = nn.Linear(num_features, n_embedding)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, 1)
        out = torch.flatten(out, 1)
        features = self.embedding(out)
        return features

def _densenet(arch, growth_rate, block_config, num_init_features, **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def densenet121(**kwargs):
    r"""3D Densenet-121 model adapted from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, **kwargs)



