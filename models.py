import torch
import torch.nn as nn
from torch.nn import functional as F


class ReLog(nn.Module):
    r"""Applies the rectified log unit function element-wise:

    :math:`\text{ReLog}(x)= \log (\max(0, x) + 1)`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    __constants__ = []

    def __init__(self):
        super(ReLog, self).__init__()

    def forward(self, input):
        return torch.log10(F.relu(input) + 0.01) + 2

    
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
                 conv1_out_channels=16, conv2_out_channels=32, use_sigmoid_out=False):
        super(CNN, self).__init__()
        self.use_maxout = use_maxout
        self.use_relog = use_relog
        self.max_folding_factor = max_folding_factor
        self.min_folding_factor = min_folding_factor
        self.folding_factor = max_folding_factor * min_folding_factor
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.use_sigmoid_out = use_sigmoid_out
        
        activation_func = ReLog if self.use_relog else nn.ReLU
        conv1_modules = self.build_conv1() + ( # input shape (1, 28, 28)
            activation_func(),                 # activation
            nn.MaxPool2d(kernel_size=2),       # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv1 = nn.Sequential(*conv1_modules)
        conv2_modules = self.build_conv2() + ( # input shape (16, 14, 14)
            activation_func(),                 # activation
            nn.MaxPool2d(2),                   # output shape (32, 7, 7)
        )
        self.conv2 = nn.Sequential(*conv2_modules)
        self.out = self.build_output()

    def build_conv1(self):
        actual_out_channels = self.conv1_out_channels * (self.folding_factor if self.use_maxout else 1)
        cnn = nn.Conv2d(
                    in_channels=1,              # input height
                    out_channels=actual_out_channels, # n_filters
                    kernel_size=5,              # filter size
                    stride=1,                   # filter movement/step
                    padding=2,                  
                )
        return self.append_maxout(cnn)
        
    def build_conv2(self):
        actual_out_channels = self.conv2_out_channels * (self.folding_factor if self.use_maxout else 1)
        cnn = nn.Conv2d(self.conv1_out_channels, actual_out_channels, 5, 1, 2)
        return self.append_maxout(cnn)
        
    def build_output(self):
        n_classes = 10 * (self.folding_factor if self.use_maxout else 1)
        out = nn.Linear(self.conv2_out_channels * 7 * 7, n_classes)
        modules = self.append_maxout(out)
        if self.use_sigmoid_out:
            modules += (nn.Sigmoid(),)
        return nn.Sequential(*modules)

    def append_maxout(self, cnn):
        if self.use_maxout == 'max':
            maxout = FoldingMaxout(self.folding_factor, dim=1)
            return (cnn, maxout)                
        elif self.use_maxout == 'min':
            minout = FoldingMaxout(self.folding_factor, dim=1, use_min=True)
            return (cnn, minout)                
        elif self.use_maxout == 'minmax':
            minout = FoldingMaxout(self.min_folding_factor, dim=1, use_min=True)
            maxout = FoldingMaxout(self.max_folding_factor, dim=1)
            return (cnn, minout, maxout)        
        else:
            return (cnn,)                       
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x # return last layer for visualization
