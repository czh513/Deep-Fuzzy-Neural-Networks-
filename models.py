import torch
import torch.nn as nn
from torch.nn import functional as F
import math

bounds = []

class DynamicInitializer(object):

    def __init__(self):
        self.run_already = False

    def __call__(self, layer, input):
        global bounds
        if not self.run_already:
            input, = input # for some reasons, it comes as a tuple
            E_x2 = (input*input).mean().item()
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in * E_x2)
            nn.init.normal_(layer.weight, mean=0, std=0.9*bound)
            nn.init.constant_(layer.bias, 0)
            bounds.append(bound)
            self.run_already = True

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

curvature_multiplier_start = 0
curvature_multiplier_stop = 1
curvature_multiplier_inc = 0.001

def update_curvature_multiplier(m):
    if m.training:
        m.multiplier = min(
            curvature_multiplier_stop, 
            m.multiplier + curvature_multiplier_inc, 
        )
    return max(0, m.multiplier)

class Spherical(nn.Linear):

    def __init__(self, *args, **kwargs):
        super(Spherical, self).__init__(*args, **kwargs)
        self.multiplier = curvature_multiplier_start

    def forward(self, input):
        output = super(Spherical, self).forward(input)
        a = 0.5 * update_curvature_multiplier(self)
        output += -a * (input*input).mean(axis=1, keepdim=True) + a
        return output

class Elliptical(nn.Linear):

    def __init__(self, *args, **kwargs):
        super(Elliptical, self).__init__(*args, **kwargs)
        kwargs['bias'] = False
        self._quadratic = nn.Linear(*args, **kwargs)
        self.multiplier = curvature_multiplier_start

    def forward(self, input):
        linear_term = super(Elliptical, self).forward(input)
        with torch.no_grad():
            self._quadratic.weight.abs_() # TODO: try clipping here, abs after initialization
        quadratic_term = self._quadratic.forward(input*input)
        a = update_curvature_multiplier(self)
        return -a * quadratic_term + a + linear_term

class SphericalCNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(SphericalCNN, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        kwargs['bias'] = False
        self.multiplier = curvature_multiplier_start
        self._cnn_mean = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        if isinstance(kernel_size, (tuple,list)):
            w, h = kernel_size
            n_elems = w * h
        else:
            n_elems = kernel_size * kernel_size
        self._cnn_mean.weight.requires_grad = False
        self._cnn_mean.weight.fill_(1/float(n_elems))

    def forward(self, input):
        output = super(SphericalCNN, self).forward(input)
        a = 0.5 * update_curvature_multiplier(self)
        output += -a * self._cnn_mean.forward(input*input) + a
        return output

class EllipticalCNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(EllipticalCNN, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        kwargs['bias'] = False
        self.multiplier = curvature_multiplier_start
        self._quadratic = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, input):
        linear_term = super(EllipticalCNN, self).forward(input)
        with torch.no_grad():
            self._quadratic.weight.abs_() # TODO: try clipping here, abs after initialization
        quadratic_term = self._quadratic.forward(input*input)
        a = update_curvature_multiplier(self)
        return -a * quadratic_term + a + linear_term

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
    

config_defaults = {
    'use_relog': False, 'use_maxout': '', 'max_folding_factor': 4, 'min_folding_factor': 2,
    'conv1_out_channels': 16, 'conv2_out_channels': 32, 'use_sigmoid_out': False, 
    'use_spherical': False, 'use_elliptical': False, 'use_batchnorm': False,
    'use_homemade_initialization': False
}

class ExperimentalModel(nn.Module):

    def __init__(self, **kwargs):
        super(ExperimentalModel, self).__init__()
        assert all(k in config_defaults for k in kwargs)
        self.conf = {**config_defaults, **kwargs}
        assert not (self.conf['use_spherical'] and self.conf['use_elliptical']), \
                "Can't use elliptical and spherical units at the same time"
        if 'max' not in self.conf['use_maxout']:
            self.conf['max_folding_factor'] = 1
        if 'min' not in self.conf['use_maxout']:
            self.conf['min_folding_factor'] = 1
        self.conf['folding_factor'] = self.conf['max_folding_factor'] * self.conf['min_folding_factor']

    def extract_weights_and_bias(self, layers):
        self.weights = []
        self.bias = []
        for layer in layers:
            if isinstance(layer, (Elliptical, EllipticalCNN)):
                self.weights.append(layer._quadratic)
            self.weights.append(layer.weight)
            self.bias.append(layer.bias)

    def dense(self, *args, **kwargs):
        if self.conf['use_spherical']:
            f = Spherical
        elif self.conf['use_elliptical']:
            f = Elliptical
        else:
            f = nn.Linear
        module = f(*args, **kwargs)
        if self.conf['use_homemade_initialization']:
            module.register_forward_pre_hook(DynamicInitializer())
        return module

    def conv(self, *args, **kwargs):
        if self.conf['use_spherical']:
            f = SphericalCNN
        elif self.conf['use_elliptical']:
            f = EllipticalCNN
        else:
            f = nn.Conv2d
        module = f(*args, **kwargs)
        if self.conf['use_homemade_initialization']:
            module.register_forward_pre_hook(DynamicInitializer())
        return module

    def activation_func(self):
        return ReLog(inplace=True) if self.conf['use_relog'] else nn.ReLU(inplace=True)

    def wrap_linear(self, linear, activ=True):
        assert not isinstance(linear, list)
        modules = [linear]
        if self.conf['use_maxout'] == 'max':
            maxout = FoldingMaxout(self.conf['folding_factor'], dim=1)
            modules.append(maxout)                
        elif self.conf['use_maxout'] == 'min':
            minout = FoldingMaxout(self.conf['folding_factor'], dim=1, use_min=True)
            modules.append(minout)                
        elif self.conf['use_maxout'] == 'minmax':
            minout = FoldingMaxout(self.conf['min_folding_factor'], dim=1, use_min=True)
            maxout = FoldingMaxout(self.conf['max_folding_factor'], dim=1)
            modules.extend([minout, maxout])
        if activ:
            modules.append(self.activation_func())
        return modules


class CNN(ExperimentalModel):

    def __init__(self, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.n_classes = 10
        # first CNN
        actual_out_channels = self.conv1_out_channels * self.conf['folding_factor']
        cnn1 = self.conv(
            in_channels=1,
            out_channels=actual_out_channels,
            kernel_size=5,              # filter size
            stride=1,                   # filter movement/step
            padding=2,                  
        )
        # second CNN
        actual_input_channels = self.conv1_out_channels
        actual_out_channels = self.conv2_out_channels * self.conf['folding_factor']
        cnn2 = self.conv(actual_input_channels, actual_out_channels, 5, stride=1, padding=2)
        # output
        actual_input_channels = self.conv2_out_channels * 7 * 7
        out = self.dense(actual_input_channels, self.n_classes * self.conf['folding_factor'])
        self.extract_weights_and_bias([self.conv1, self.conv2, self.out])

        self.features = nn.Sequential(
            self.wrap_linear(cnn1) + [nn.MaxPool2d(kernel_size=2)]
            + self.wrap_linear(cnn2) + [nn.MaxPool2d(kernel_size=2)]
        )
        self.out = nn.Sequential(
            self.wrap_linear(out, activ=False) 
            + ([nn.Sigmoid()] if self.conf['use_sigmoid_out'] else [])
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x # return last layer for visualization


cfg = {
    'VGG3':  [64, 'M', 128, 'M', 256, 'M', 'M', 'M'], # a scale-downed version to test my code
    'VGG4':  [64, 'M', 128, 'M', 256, 256, 'M', 'M', 'M'], # a scale-downed version to test my code
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(ExperimentalModel):

    def __init__(self, vgg_name="VGG11", **kwargs):
        super(VGG, self).__init__(**kwargs)
        self.n_classes = 10
        self.features, conv_layers, last_layer_size = self._make_layers(cfg[vgg_name])
        classifier = self.dense(last_layer_size, 10 * self.conf['folding_factor'])
        self.classifier = nn.Sequential(*self.wrap_linear(classifier, activ=False))
        self.extract_weights_and_bias(conv_layers + [classifier])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        if self.conf['use_sigmoid_out']:
            out = nn.Sigmoid()(out)
        return out

    def _make_layers(self, cfg):
        layers, conv_layers = [], []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(x, int):
                conv = self.conv(in_channels, x * self.conf['folding_factor'], kernel_size=3, padding=1)
                conv_layers.append(conv)
                layers += self.wrap_linear(conv)
                if self.conf['use_batchnorm']:
                    layers += [nn.BatchNorm2d(x * self.conf['folding_factor'])]
                in_channels = x
            else:
                raise "Unrecognized config token: %s" % str(x)
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers), conv_layers, in_channels
