import torch
import torch.nn as nn
from torch.nn import functional as F
import math

bounds = []

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

# put it to negative if you want to start after some epochs
log_strength_start = 0
log_strength_inc = 0.001
log_strength_stop = 1

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
        self.log_strength = log_strength_start

    def forward(self, input):
        if self.training:
            self.log_strength = min(log_strength_stop, self.log_strength + log_strength_inc)
        beta = max(1e-4, self.log_strength) # effective log strength
        relog_func = lambda x: torch.log(F.relu(x)*beta + 1) / beta
        return relog_func(input)

    def extra_repr(self):
        return 'n=%.2f' % (self.n)

# if you want a gradual ramping up, change this
# put it to negative if you want to start after some epochs
curvature_multiplier_start = 0
curvature_multiplier_inc = 0.001
curvature_multiplier_stop = 1

def update_curvature_multiplier(m):
    if m.training:
        m.multiplier = min(
            curvature_multiplier_stop, 
            m.multiplier + curvature_multiplier_inc)
    return max(0, m.multiplier)

class AbsLinear(nn.Linear):
    ''' A linear module that always applies abs() on the weight '''
    def forward(self, input):
        return F.linear(input, self.weight.abs(), self.bias)

class AbsConv2d(nn.Conv2d):
    ''' A convolutional module that always applies abs() on the weight '''
    def forward(self, input):
        return self.conv2d_forward(input, self.weight.abs())

class Elliptical(nn.Linear):

    def __init__(self, *args, **kwargs):
        super(Elliptical, self).__init__(*args, **kwargs)
        kwargs['bias'] = False
        self.multiplier = curvature_multiplier_start
        self._quadratic = AbsLinear(*args, **kwargs)

    def forward(self, input):
        linear_term = super(Elliptical, self).forward(input)
        quadratic_term = self._quadratic.forward(input*input)
        a = update_curvature_multiplier(self)
        fan_in = self.weight.shape[1]
        return -a * quadratic_term + a * math.sqrt(fan_in) + linear_term

class Quadratic(Elliptical):

    def __init__(self, *args, **kwargs):
        super(Quadratic, self).__init__(*args, **kwargs)
        self._quadratic = nn.Linear(*args, **kwargs)

class EllipticalCNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(EllipticalCNN, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        kwargs['bias'] = False
        self.multiplier = curvature_multiplier_start
        self._quadratic = AbsConv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, input):
        linear_term = super(EllipticalCNN, self).forward(input)
        quadratic_term = self._quadratic.forward(input*input)
        a = update_curvature_multiplier(self)
        fan_in = self.weight.shape[1]
        return -a * quadratic_term + a * math.sqrt(fan_in) + linear_term

class QuadraticCNN(EllipticalCNN):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(QuadraticCNN, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._quadratic = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

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
    
class ExperimentalModel(nn.Module):

    def __init__(self, **kwargs):
        super(ExperimentalModel, self).__init__()
        assert all(k in self.config_defaults for k in kwargs)
        self.conf = {**self.config_defaults, **kwargs}
        print('Using model config:', self.conf)
        assert sum([self.conf['use_elliptical'], self.conf['use_quadratic']]) in (0, 1), \
                "Can only use one in elliptical and quadratic"
        if 'max' not in self.conf['use_maxout']:
            self.conf['max_folding_factor'] = 1
        if 'min' not in self.conf['use_maxout']:
            self.conf['min_folding_factor'] = 1
        self.conf['folding_factor'] = self.conf['max_folding_factor'] * self.conf['min_folding_factor']

    def extract_weights_and_bias(self, layers):
        self.weights = []
        self.bias = []
        for layer in layers:
            if hasattr(layer, '_quadratic'):
                self.weights.append(layer._quadratic.weight)
            self.weights.append(layer.weight)
            self.bias.append(layer.bias)

    def quadratic_weights(self):
        weights = []
        for layer in list(self.features) + list(self.classifier):
            if hasattr(layer, '_quadratic'):
                weights.append(layer._quadratic.weight)
        return weights
                
    def dense(self, *args, **kwargs):
        if self.conf['use_elliptical']:
            f = Elliptical
        elif self.conf['use_quadratic']:
            f = Quadratic
        else:
            f = nn.Linear
        module = f(*args, **kwargs)
        return module

    def conv(self, *args, **kwargs):
        if self.conf['use_elliptical']:
            f = EllipticalCNN
        elif self.conf['use_quadratic']:
            f = QuadraticCNN
        else:
            f = nn.Conv2d
        module = f(*args, **kwargs)
        return module

    def activation_func(self, layer_no=None):
        return (ReLog(self.conf['relog_n'], inplace=True) if self.conf['use_relog']
                else nn.ReLU(inplace=True))

    def wrap_linear(self, linear, activ=True, layer_no=None):
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
            modules.append(self.activation_func(layer_no=layer_no))
        return modules

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out, x


class CNN(ExperimentalModel):

    config_defaults = {
        'use_relog': False, 'relog_n': 10,
        'use_maxout': '', 'max_folding_factor': 4, 'min_folding_factor': 2,
        'conv1_out_channels': 16, 'conv2_out_channels': 32,
        'use_elliptical': False, 'use_quadratic': False, 'use_batchnorm': False,
    }

    def __init__(self, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.n_classes = 10
        # first CNN
        actual_out_channels = self.conf['conv1_out_channels'] * self.conf['folding_factor']
        cnn1 = self.conv(
            in_channels=1,
            out_channels=actual_out_channels,
            kernel_size=5,              # filter size
            stride=1,                   # filter movement/step
            padding=2,                  
        )
        # second CNN
        actual_input_channels = self.conf['conv1_out_channels']
        actual_out_channels = self.conf['conv2_out_channels'] * self.conf['folding_factor']
        cnn2 = self.conv(actual_input_channels, actual_out_channels, 5, stride=1, padding=2)
        # output
        actual_input_channels = self.conf['conv2_out_channels'] * 7 * 7
        out = self.dense(actual_input_channels, self.n_classes * self.conf['folding_factor'])
        self.extract_weights_and_bias([cnn1, cnn2, out])

        self.features = nn.Sequential(*(
            self.wrap_linear(cnn1) + [nn.MaxPool2d(kernel_size=2)]
            + self.wrap_linear(cnn2) + [nn.MaxPool2d(kernel_size=2)]
        ))
        self.classifier = nn.Sequential(*self.wrap_linear(out, activ=False))


cfg = {
    'VGG3':  [64, 'M', 128, 'M', 256, 'M', 'M', 'M'], # a scale-downed version to test my code
    'VGG4':  [64, 'M', 128, 'M', 256, 256, 'M', 'M', 'M'], # a scale-downed version to test my code
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # a scale-downed version to test my code
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(ExperimentalModel):

    config_defaults = {
        'use_relog': False, 'relog_n': 5, 'modification_start_layer': 0, 
        'use_maxout': '', 'max_folding_factor': 4, 'min_folding_factor': 2,
        'use_elliptical': False, 'use_quadratic': False, 'use_batchnorm': False,
        'vgg_name': 'VGG16', 'capacity_factor': 1
    }

    def __init__(self, vgg_name="VGG11", **kwargs):
        super(VGG, self).__init__(**kwargs)
        self.n_classes = 10
        self.features, conv_layers, last_layer_size = self._make_layers(cfg[vgg_name])
        classifier = self.dense(last_layer_size, 10 * self.conf['folding_factor'])
        self.classifier = nn.Sequential(*self.wrap_linear(classifier, activ=False))
        self.extract_weights_and_bias(conv_layers + [classifier])

    def _make_layers(self, cfg):
        layers, conv_layers = [], []
        in_channels = 3
        for layer_no, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(x, int):
                if layer_no >= self.conf['modification_start_layer']:
                    x = int(x*self.conf['capacity_factor'])
                    conv = self.conv(in_channels, x * self.conf['folding_factor'], kernel_size=3, padding=1)
                    layers += self.wrap_linear(conv, layer_no=layer_no)
                else:
                    conv = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                    layers += [conv, nn.ReLU()]
                if self.conf['use_batchnorm']:
                    # set track_running_stats=True and use a small momentum value 
                    # here (which actually mean _more_ stability)
                    # so that our use of ReLog is still meaningful. 
                    layers += [nn.BatchNorm2d(x, momentum=0.01, track_running_stats=True)]
                in_channels = x
                conv_layers.append(conv)
            else:
                raise "Unrecognized config token: %s" % str(x)
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers), conv_layers, in_channels

def extract_curvature_strength(model):
    multipliers = [layer.multiplier for layer in model.features
                   if hasattr(layer, 'multiplier')]
    if len(multipliers) > 0:
        multiplier, = list(set(multipliers)) # make sure they're the same
        return multiplier
    else:
        return float('nan')

def extract_log_strength(model):
    log_strengths = [layer.log_strength for layer in model.features
                   if hasattr(layer, 'log_strength')]
    if len(log_strengths) > 0:
        log_strength, = list(set(log_strengths)) # make sure they're the same
        return log_strength
    else:
        return float('nan')