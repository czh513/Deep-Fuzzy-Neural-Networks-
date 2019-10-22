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
    

class CNN(nn.Module):

    def __init__(self, use_relog=False, use_maxout='', max_folding_factor=4, min_folding_factor=2,
                 conv1_out_channels=16, conv2_out_channels=32, use_sigmoid_out=False, dropout_prob=0,
                 use_spherical=False, use_elliptical=False):
        assert not (use_spherical and use_elliptical), "Can't use elliptical and spherical units at the same time"
        super(CNN, self).__init__()
        self.use_maxout = use_maxout
        self.use_relog = use_relog
        self.max_folding_factor = max_folding_factor if 'max' in use_maxout else 1
        self.min_folding_factor = min_folding_factor if 'min' in use_maxout else 1
        self.folding_factor = self.max_folding_factor * self.min_folding_factor
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.use_sigmoid_out = use_sigmoid_out
        self.n_classes = 10
        self.dropout_prob = dropout_prob
        self.use_spherical = use_spherical
        self.use_elliptical = use_elliptical
        self.weights = []
        self.bias = []

        conv1_modules = self.build_conv1() + [nn.MaxPool2d(kernel_size=2)]
        self.conv1 = nn.Sequential(*conv1_modules)
        conv2_modules = self.build_conv2() + [nn.MaxPool2d(2)]
        self.conv2 = nn.Sequential(*conv2_modules)
        self.out = self.build_output()

    def dense(self, *args, **kwargs):
        if self.use_spherical:
            f = Spherical
        elif self.use_elliptical:
            f = Elliptical
        else:
            f = nn.Linear
        return f(*args, **kwargs)

    def conv(self, *args, **kwargs):
        if self.use_spherical:
            f = SphericalCNN
        elif self.use_elliptical:
            f = EllipticalCNN
        else:
            f = nn.Conv2d
        return f(*args, **kwargs)

    def activation_func(self):
        return ReLog(inplace=True) if self.use_relog else nn.ReLU(inplace=True)

    def dropout(self):
        return nn.Dropout(self.dropout_prob, inplace=True)

    def build_conv1(self):
        actual_out_channels = self.conv1_out_channels * self.folding_factor
        cnn = self.conv(
                    in_channels=1,
                    out_channels=actual_out_channels,
                    kernel_size=5,              # filter size
                    stride=1,                   # filter movement/step
                    padding=2,                  
                )
        self.weights.append(cnn.weight)
        self.bias.append(cnn.bias)
        return self.wrap_linear(cnn)
        
    def build_conv2(self):
        actual_input_channels = self.conv1_out_channels
        actual_out_channels = self.conv2_out_channels * self.folding_factor
        cnn = self.conv(actual_input_channels, actual_out_channels, 5, stride=1, padding=2)
        self.weights.append(cnn.weight)
        self.bias.append(cnn.bias)
        return self.wrap_linear(cnn)
        
    def build_output(self):
        actual_input_channels = self.conv2_out_channels * 7 * 7
        out = self.dense(actual_input_channels, self.n_classes * self.folding_factor)
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
            modules.append(self.activation_func())
        return modules

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
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

class VGG(nn.Module):

    def __init__(self, vgg_name="VGG11", use_relog=False, relog_n=10, use_maxout=None, max_folding_factor=4, 
                 min_folding_factor=1, use_sigmoid_out=False, use_batchnorm=True, use_homemade_initialization=False,
                 use_spherical=False, use_elliptical=False):
        super(VGG, self).__init__()
        assert not (use_spherical and use_elliptical), "Can't use both spherical and elliptical at the same time"
        self.use_maxout = use_maxout
        self.use_relog = use_relog
        self.relog_n = relog_n
        self.max_folding_factor = max_folding_factor
        self.min_folding_factor = min_folding_factor
        self.folding_factor = max_folding_factor * min_folding_factor if self.use_maxout else 1
        self.use_sigmoid_out = use_sigmoid_out
        self.use_batchnorm = use_batchnorm
        self.use_homemade_initialization = use_homemade_initialization
        self.use_spherical = use_spherical
        self.use_elliptical = use_elliptical

        self.n_classes = 10
        self.weights = []
        self.bias = []
        self.features, last_layer_size = self._make_layers(cfg[vgg_name])
        classifier = self.dense(last_layer_size, 10 * self.folding_factor)
        self.weights.append(classifier.weight)
        self.bias.append(classifier.bias)
        classifier = self.append_maxout([classifier])
        self.classifier = nn.Sequential(*classifier)

    def dense(self, *args, **kwargs):
        if self.use_spherical:
            f = Spherical
        elif self.use_elliptical:
            f = Elliptical
        else:
            f = nn.Linear
        module = f(*args, **kwargs)
        if self.use_homemade_initialization:
            module.register_forward_pre_hook(DynamicInitializer())
        return module

    def conv(self, *args, **kwargs):
        if self.use_spherical:
            f = SphericalCNN
        elif self.use_elliptical:
            f = EllipticalCNN
        else:
            f = nn.Conv2d
        module = f(*args, **kwargs)
        if self.use_homemade_initialization:
            module.register_forward_pre_hook(DynamicInitializer())
        return module

    def activation_func(self):
        return ReLog(self.relog_n, inplace=True) if self.use_relog else nn.ReLU(inplace=True)

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
            elif isinstance(x, int):
                layers += [self.conv(in_channels, 
                                          x * self.folding_factor, kernel_size=3, padding=1)]
                if self.use_batchnorm:
                    layers += [nn.BatchNorm2d(x * self.folding_factor)]
                layers += [self.activation_func()]
                self.append_maxout(layers)
                in_channels = x
            else:
                raise "Unrecognized config token: %s" % str(x)
            self.weights.extend(m.weight for m in layers if hasattr(m, 'weight'))
            self.bias.extend(m.bias for m in layers if hasattr(m, 'bias'))
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers), in_channels

    def append_maxout(self, layers):
        if self.use_maxout == 'max':
            layers += [FoldingMaxout(self.max_folding_factor, dim=1)]
        elif self.use_maxout == 'min':
            layers += [FoldingMaxout(self.min_folding_factor, dim=1, use_min=True)]
        elif self.use_maxout == 'minmax':
            layers += [
                FoldingMaxout(self.min_folding_factor, dim=1, use_min=True),
                FoldingMaxout(self.max_folding_factor, dim=1)
            ] 
        return layers       