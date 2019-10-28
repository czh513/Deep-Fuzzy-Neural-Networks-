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

log_strength_start = 0
log_strength_inc = 0.001

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
            self.log_strength = min(1, self.log_strength + log_strength_inc)
        effective_log_strength = max(0, self.log_strength)
        linear_term = F.relu(input)
        log_term = torch.log(linear_term + 1/self.n) / math.log(self.n) + 1
        if effective_log_strength < 1:
            # interpolate ReLog-ReLU just in case training is unstable
            return log_term * effective_log_strength + linear_term * (1-effective_log_strength)
        else:
            return log_term

    def extra_repr(self):
        return 'n=%.2f' % (self.n)

curvature_multiplier_start = 0
curvature_multiplier_inc = 0.001

def update_curvature_multiplier(m):
    if m.training:
        m.multiplier = min(1, m.multiplier + curvature_multiplier_inc)
    return max(0, m.multiplier)

class AbsLinear(nn.Linear):
    ''' A linear module that always applies abs() on the weight '''
    def forward(self, input):
        return F.linear(input, self.weight.abs(), self.bias)

class AbsConv2d(nn.Conv2d):
    ''' A convolutional module that always applies abs() on the weight '''
    def forward(self, input):
        return self.conv2d_forward(input, self.weight.abs())

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
        self.multiplier = curvature_multiplier_start
        self._quadratic = AbsLinear(*args, **kwargs)

    def forward(self, input):
        linear_term = super(Elliptical, self).forward(input)
        quadratic_term = self._quadratic.forward(input*input)
        a = update_curvature_multiplier(self)
        return -a * quadratic_term + a + linear_term

class Quadratic(Elliptical):

    def __init__(self, *args, **kwargs):
        super(Quadratic, self).__init__(*args, **kwargs)
        self._quadratic = nn.Linear(*args, **kwargs)

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
        self._quadratic = AbsConv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, input):
        linear_term = super(EllipticalCNN, self).forward(input)
        quadratic_term = self._quadratic.forward(input*input)
        a = update_curvature_multiplier(self)
        return -a * quadratic_term + a + linear_term

class QuadraticCNN(nn.Conv2d):

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
    
config_defaults = {
    'use_relog': False, 'modification_start_layer': 0, 'use_maxout': '', 'max_folding_factor': 4, 'min_folding_factor': 2,
    'conv1_out_channels': 16, 'conv2_out_channels': 32, 'use_sigmoid_out': False, 
    'use_spherical': False, 'use_elliptical': False, 'use_quadratic': False, 'use_batchnorm': False,
    'use_homemade_initialization': False, 'vgg_name': 'VGG16'
}

class ExperimentalModel(nn.Module):

    def __init__(self, **kwargs):
        super(ExperimentalModel, self).__init__()
        assert all(k in config_defaults for k in kwargs)
        self.conf = {**config_defaults, **kwargs}
        assert sum([self.conf['use_spherical'], self.conf['use_elliptical'], 
                    self.conf['use_quadratic']]) in (0, 1), \
                "Can only use one in spherical, elliptical, and quadratic"
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
        if self.conf['use_spherical']:
            f = Spherical
        elif self.conf['use_elliptical']:
            f = Elliptical
        elif self.conf['use_quadratic']:
            f = Quadratic
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
        elif self.conf['use_quadratic']:
            f = QuadraticCNN
        else:
            f = nn.Conv2d
        module = f(*args, **kwargs)
        if self.conf['use_homemade_initialization']:
            module.register_forward_pre_hook(DynamicInitializer())
        return module

    def activation_func(self, layer_no=None):
        return (ReLog(inplace=True) if self.conf['use_relog']
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
        self.classifier = nn.Sequential(*(
            self.wrap_linear(out, activ=False) 
            + ([nn.Sigmoid()] if self.conf['use_sigmoid_out'] else [])
        ))


cfg = {
    'VGG3':  [64, 'M', 128, 'M', 256, 'M', 'M', 'M'], # a scale-downed version to test my code
    'VGG4':  [64, 'M', 128, 'M', 256, 256, 'M', 'M', 'M'], # a scale-downed version to test my code
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # a scale-downed version to test my code
    'VGG11*': [1024, 'M', 512, 'M', 512, 512, 'M', 256, 256, 'M', 128, 128, 'M'], # a scale-downed version to test my code
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16*': [512, 512, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(ExperimentalModel):

    def __init__(self, vgg_name="VGG11", **kwargs):
        super(VGG, self).__init__(**kwargs)
        self.n_classes = 10
        self.features, conv_layers, last_layer_size = self._make_layers(cfg[vgg_name])
        classifier = self.dense(last_layer_size, 10 * self.conf['folding_factor'])
        self.classifier = nn.Sequential(*(
            self.wrap_linear(classifier, activ=False)
            + ([nn.Sigmoid()] if self.conf['use_sigmoid_out'] else [])
        ))
        self.extract_weights_and_bias(conv_layers + [classifier])

    def _make_layers(self, cfg):
        layers, conv_layers = [], []
        in_channels = 3
        for layer_no, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(x, int):
                if layer_no >= self.conf['modification_start_layer']:
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
