import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class ReLog(nn.Module):
    r"""Applies the rectified log unit function element-wise:

    :math:`\text{ReLog}(x)= \log (\max(0, x) + 1)`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    __constants__ = []

    def __init__(self, n=10, inplace=False):
        assert(n > 1)
        super(ReLog, self).__init__()
        self.n = n
        self.inplace = inplace

    def forward(self, input):
        # can't have two subsequent in-place operations, otherwise will get error:
        # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        return torch.log(F.relu(input, self.inplace) + 1/self.n) / math.log(self.n) + 1

    def extra_repr(self):
        return 'n=%.2f' % (self.n)

    
class FoldingMaxout(nn.Module):
    ''' Fold the previous layer into k-tuples and output the max of each.
    
    The dimension being folded is controlled by `dim` parameter, the output is `k` times
    less elements along `dim` than the input.
    '''

    def __init__(self, k, dim, use_min=False):
        super().__init__()
        self.k = k
        self.dim = dim
        self.use_min = use_min

    def forward(self, input):
        s, d, k = input.shape, self.dim, self.k
        assert s[d] % k == 0
        new_shape = s[:d] + (s[d]//k, k) + s[d+1:]
        folded = input.reshape(new_shape)
        output, _ = folded.min(d+1) if self.use_min else folded.max(d+1)
        return output

    def extra_repr(self):
        return 'use_min=%s, k=%s' % (self.use_min, self.k)
    

class CNN(nn.Module):

    def __init__(self, use_relog=False, use_maxout=None, max_folding_factor=4, min_folding_factor=1,
                 conv1_out_channels=16, conv2_out_channels=32, use_sigmoid_out=False, dropout_prob=0.15):
        super(CNN, self).__init__()
        self.use_maxout = use_maxout
        self.use_relog = use_relog
        self.max_folding_factor = max_folding_factor
        self.min_folding_factor = min_folding_factor
        self.folding_factor = max_folding_factor * min_folding_factor if self.use_maxout else 1
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.use_sigmoid_out = use_sigmoid_out
        self.n_classes = 10
        self.activation_func = ReLog if self.use_relog else nn.ReLU
        self.dropout_prob = dropout_prob
        self.weights = []
        self.bias = []

        conv1_modules = self.build_conv1() + [nn.MaxPool2d(kernel_size=2)]
        self.conv1 = nn.Sequential(*conv1_modules)
        conv2_modules = self.build_conv2() + [nn.MaxPool2d(2)]
        self.conv2 = nn.Sequential(*conv2_modules)
        self.out = self.build_output()

    def dropout(self):
        return nn.Dropout(self.dropout_prob, inplace=True)

    def build_conv1(self):
        actual_out_channels = self.conv1_out_channels * self.folding_factor
        cnn = nn.Conv2d(
                    in_channels=1,              # input height
                    out_channels=actual_out_channels, # n_filters
                    kernel_size=5,              # filter size
                    stride=1,                   # filter movement/step
                    padding=2,                  
                )
        self.weights.append(cnn.weight)
        self.bias.append(cnn.bias)
        return self.wrap_linear(cnn)
        
    def build_conv2(self):
        actual_out_channels = self.conv2_out_channels * self.folding_factor
        cnn = nn.Conv2d(self.conv1_out_channels, actual_out_channels, 5, 1, 2)
        self.weights.append(cnn.weight)
        self.bias.append(cnn.bias)
        return self.wrap_linear(cnn)
        
    def build_output(self):
        out = nn.Linear(self.conv2_out_channels * 7 * 7, self.n_classes * self.folding_factor)
        self.weights.append(out.weight)
        self.bias.append(out.bias)
        modules = self.wrap_linear(out, activ=False)
        if self.use_sigmoid_out:
            modules += (nn.Sigmoid(),)
        return nn.Sequential(*modules)

    def wrap_linear(self, linear, activ=True):
        assert not isinstance(linear, list)
        modules = [linear]
        if self.dropout_prob > 0:
            modules.insert(0, self.dropout())
        if self.use_maxout == 'max':
            maxout = FoldingMaxout(self.folding_factor, dim=1)
            modules.append(maxout)                
        elif self.use_maxout == 'min':
            minout = FoldingMaxout(self.folding_factor, dim=1, use_min=True)
            modules.append(minout)                
        elif self.use_maxout == 'minmax':
            minout = FoldingMaxout(self.min_folding_factor, dim=1, use_min=True)
            maxout = FoldingMaxout(self.max_folding_factor, dim=1)
            modules.extend([minout, maxout])
        if activ:
            modules.append(self.activation_func(inplace=True))
        return modules

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x # return last layer for visualization


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name="VGG11", use_relog=False, relog_n=10, use_maxout=None, max_folding_factor=4, 
                 min_folding_factor=1, use_sigmoid_out=False, dropout_prob=0):
        super(VGG, self).__init__()
        self.use_maxout = use_maxout
        self.use_relog = use_relog
        self.relog_n = relog_n
        self.max_folding_factor = max_folding_factor
        self.min_folding_factor = min_folding_factor
        self.folding_factor = max_folding_factor * min_folding_factor if self.use_maxout else 1
        self.use_sigmoid_out = use_sigmoid_out
        self.n_classes = 10
        self.activation_func = (lambda: ReLog(relog_n)) if self.use_relog else nn.ReLU
        self.dropout_prob = dropout_prob
        self.weights = []
        self.bias = []
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        if self.use_sigmoid_out:
            out = nn.Sigmoid()(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.dropout_prob and self.dropout_prob > 0:
                    layers += [nn.Dropout(self.dropout_prob)]
                layers += [nn.Conv2d(in_channels, x * self.folding_factor, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x * self.folding_factor)]
                if self.use_maxout == 'max':
                    layers += [FoldingMaxout(self.max_folding_factor, dim=1)]
                elif self.use_maxout == 'min':
                    layers += [FoldingMaxout(self.min_folding_factor, dim=1, use_min=True)]
                elif self.use_maxout == 'minmax':
                    layers += [
                        FoldingMaxout(self.min_folding_factor, dim=1, use_min=True),
                        FoldingMaxout(self.max_folding_factor, dim=1)
                    ]
                layers += [self.activation_func(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

