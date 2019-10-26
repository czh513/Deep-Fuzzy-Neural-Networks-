import os
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import models
from utils import grouper_variable_length
from time import time
import torch.optim as optim
import itertools
import math
import numpy as np

def gaussian_noise(epsilon):
    transform_func = lambda x : (x + torch.randn_like(x)*epsilon).clamp(min=0, max=1)
    return torchvision.transforms.Lambda(transform_func)

class PixelScramble(object):

    def __call__(self, input):
        ''' Scramble pixels in a picture, without crossing color channels '''
        output = input.view(input.shape[0], input.shape[1], -1)
        order = torch.randperm(output.shape[2])
        return output[:,:,order].view(input.shape)

class BlockScramble(object):
    
    def __init__(self, block_size=5):
        self.block_size = block_size

    def blocked_randperm(self, n):
      n_round = int(math.ceil(n / float(self.block_size)) * self.block_size)
      indices = torch.arange(n_round).fmod_(n)
      blocked_indices = indices.reshape(-1, self.block_size)
      blocked_indices = blocked_indices[torch.randperm(blocked_indices.shape[0])]
      indices = blocked_indices.view(-1)[:n]
      return indices
      
    def __call__(self, input):
        ''' Scramble pixels in a picture, without crossing color channels '''
        output = (input
                  [:,:,self.blocked_randperm(input.shape[2]),:]
                  [:,:,:,self.blocked_randperm(input.shape[3])])
        return output

class ChoiceScramble(object):

    def __init__(self, scrambles):
        self.scrambles = scrambles

    def __call__(self, input):
        i = np.random.choice(len(self.scrambles))
        return self.scrambles[i](input)

def tmean(vals):
    ''' Compute micro-average of a list of Torch Tensors '''
    vals = [val.view(-1) for val in vals]
    return torch.cat(vals).mean()

class TrainingService(object):

    def compute_loss(self, epoch, cnn, train_x, train_y, output, conf):
        loss_func = nn.MSELoss() if conf['use_sigmoid_out'] else nn.CrossEntropyLoss()
        if conf['use_sigmoid_out']:
            train_y = one_hot(train_y, num_classes=self.num_classes).float()
        loss = loss_func(output, train_y)
        if conf['use_scrambling'] and epoch >= 1:
            assert conf['use_sigmoid_out'], "Softmax networks can't handle scrambled input"
            scramble_x = self.scramble(train_x)[:train_x.shape[0]]
            scramble_y = torch.zeros(scramble_x.shape[0], self.num_classes).to(self.device)
            scramble_output, _ = cnn(scramble_x)
            loss += loss_func(scramble_output, scramble_y)
        if conf['regularization'] in ('max-fit', 'max-margin') and not conf['use_spherical']:
            assert (conf['l1'] > 0) or (conf['l2'] > 0), "Strength of regularization must be specified"
            if conf['l1'] > 0:
                loss += conf['l1'] * tmean(w.abs() for w in cnn.weights)
            if conf['l2'] > 0:
                loss += conf['l2'] * tmean(w*w for w in cnn.weights)
        if conf['regularization'] == 'max-margin' and conf['use_spherical']:
            assert (conf['bias_l1'] > 0) or (conf['bias_l2'] > 0), \
                "For max-margin with spherical, strength of bias regularization must be specified"
            if conf['bias_l1'] > 0:
                loss += -conf['bias_l1'] * tmean(b for b in cnn.bias) # notice: no abs()
            if conf['bias_l2'] > 0:
                loss += -conf['bias_l2'] * tmean(b*b.abs() for b in cnn.bias) # notice: signed
        if conf['regularization'] == 'max-fit':
            assert (conf['bias_l1'] > 0) or (conf['bias_l2'] > 0), \
                "For max-fit, strength of bias regularization must be specified"
            if conf['bias_l1'] > 0:
                loss += conf['bias_l1'] * tmean(b for b in cnn.bias) # notice: no abs()
            if conf['bias_l2'] > 0:
                loss += conf['bias_l2'] * tmean(b*b.abs() for b in cnn.bias) # notice: signed
        return loss


class TrainingService_MNIST(TrainingService):

    def __init__(self, home_dir, gaussian_noise_epsilon=0.3, device='cpu'): 
        self.home_dir = home_dir
        self.load_data(gaussian_noise_epsilon)
        self.num_classes = 10
        self.scramble = ChoiceScramble([PixelScramble(), BlockScramble(5), BlockScramble(7)])
        self.device = device

    def load_data(self, gaussian_noise_epsilon):
        self.data_dir = os.path.join(self.home_dir, 'mnist')
        download_data = not(os.path.exists(self.data_dir)) or not os.listdir(self.data_dir)
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 1.0)),
            torchvision.transforms.RandomRotation(degrees=10),
            torchvision.transforms.ToTensor(),
            gaussian_noise(gaussian_noise_epsilon),
            torchvision.transforms.RandomErasing(scale=(0.01, 0.05))
        ])
        self.train_data = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=train_transform, 
            download=download_data,
        )
        self.test_data = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor()
        )

    def build_and_train(self, n_epochs=20, **kwargs):
        model_kwargs = {k:v for k, v in kwargs.items() if k in models.config_defaults}
        cnn = models.CNN(**model_kwargs)
        print(cnn)  # net architecture

        config_defaults = {
            'use_sigmoid_out': False, 'lr': 0.001,
            'train_batch_size': 64, 'regularization': None, 'l1': 0, 'l2': 0, 
            'bias_l1': 0, 'bias_l2': 0, 'use_scrambling': False,
            'use_spherical': False, 'use_elliptical': False, 'use_quadratic': False
        }
        train_params = {**config_defaults, **kwargs}
        self.train_loop(cnn, n_epochs, train_params)
        return cnn

    def train_loop(self, cnn, n_epochs, conf):
        train_loader = Data.DataLoader(dataset=self.train_data, batch_size=conf['train_batch_size'], shuffle=True)
        test_loader = Data.DataLoader(dataset=self.test_data, batch_size=2000)
        optimizer = torch.optim.Adam(cnn.parameters(), amsgrad=True, lr=conf['lr'])

        cnn.to(self.device)
        started_sec = time()
        correct, total = 0, 0
        for epoch in range(n_epochs):
            for batch_no, (train_x, train_y) in enumerate(train_loader):
                cnn.train()
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                output, _ = cnn(train_x)

                optimizer.zero_grad()           # clear gradients for this training step
                loss = self.compute_loss(epoch, cnn, train_x, train_y, output, conf)
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients

                _, pred_y = torch.max(output, dim=1)
                correct += (pred_y == train_y).sum().item()
                total += train_y.size(0)

                if batch_no % 400 == 0:
                    print('Epoch: %d | batch: %d | train acc: %.2f (%d / %d)' 
                          % (epoch, batch_no, correct/total, correct, total))
            train_acc = correct / total

            with torch.no_grad():
                cnn.eval()
                correct, total = 0, 0
                for test_x, test_y in test_loader:
                    test_x = test_x.to(self.device)
                    test_y = test_y.to(self.device)
                    test_output, _ = cnn(test_x)
                    _, pred_y = torch.max(test_output, dim=1)
                    correct += (pred_y == test_y).sum().item()
                    total += test_y.shape[0]
                test_acc = correct / total
                print('Epoch: %d | train loss: %.4f | train acc: %.2f | test acc: %.2f' 
                      % (epoch, loss.item(), train_acc, test_acc))


class TrainingService_CIFAR10(TrainingService):

    def __init__(self, home_dir, normalize_data, device='cpu', gaussian_noise_epsilon=0.3,
                 num_batches_per_epoch=-1, report_interval=150):
        self.home_dir = home_dir
        self.prepare_data(normalize_data, gaussian_noise_epsilon)
        self.device = device
        self.num_batches_per_epoch = num_batches_per_epoch # set to a small value for testing
        self.report_interval = report_interval
        self.scramble = ChoiceScramble([PixelScramble(), BlockScramble(3), BlockScramble(9)])
        self.num_classes = 10

    def prepare_data(self, normalize_data, gaussian_noise_epsilon):
        # Data
        print('==> Preparing data..')
        cifar_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # Tried random rotating, gaussian noise, and random erasing as in MNIST 
        # but models fail to learn. Didn't try them individually.
        transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
        ] + ([torchvision.transforms.Normalize(*cifar_stats)] if normalize_data else []))
        transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
        ] + ([torchvision.transforms.Normalize(*cifar_stats)] if normalize_data else []))
        self.trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train)
        self.testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform_test)

    # Training
    def train(self, net, epoch, conf):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, _ = net(inputs)
            loss = self.compute_loss(epoch, net, inputs, targets, outputs, conf)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct/total
            if batch_idx % self.report_interval == 0:
                print(batch_idx, '/', len(self.trainloader), ': Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*acc, correct, total))
            if self.num_batches_per_epoch > 0 and batch_idx >= self.num_batches_per_epoch:
                break # end it early so that we can test the code
        return acc

    def test(self, net, epoch, conf):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _ = net(inputs)

                if conf['use_sigmoid_out']:
                    loss_targets = one_hot(targets, num_classes=10).float()
                    loss_func = nn.MSELoss()
                else:
                    loss_targets = targets
                    loss_func = nn.CrossEntropyLoss()
                loss = loss_func(outputs, loss_targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = correct/total
        print('Test eval: Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*acc, correct, total))
        return acc

    def build_and_train(self, curvature_multiplier_inc=1e-4, **kwargs):
        config_defaults = {
            'use_sigmoid_out': False, 'lr': 0.01,
            'train_batch_size': 128, 'regularization': None, 'l1': 0, 'l2': 0, 
            'bias_l1': 0, 'bias_l2': 0, 'use_scrambling': False,
            'use_spherical': False, 'use_elliptical': False, 'use_quadratic': False, 
            'use_homemade_initialization': False
        }
        unrecognized_params = [k for k in kwargs
                               if not (k in models.config_defaults or k in config_defaults)]
        assert not unrecognized_params, 'Unrecognized parameter: ' + str(unrecognized_params) 

        model_kwargs = {k:v for k, v in kwargs.items() if k in models.config_defaults}
        net = models.VGG(**model_kwargs)
        print(net)

        conf = {**config_defaults, **kwargs}
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=conf['train_batch_size'], shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=256, shuffle=False, num_workers=2)
        # tried ADAM already: it works for ReLU but fail to train ReLog (it doesn't just overfit,
        # it increases the loss after a few epochs)
        models.curvature_multiplier_inc = curvature_multiplier_inc
        net = net.to(self.device)
        lr_schedule = [0.1]*20 + [0.01]*10 + [0.001]*10
        for epoch, lr in enumerate(lr_schedule):
            self.optimizer = optim.SGD(net.parameters(), lr=conf['lr'])
            last_train_acc = self.train(net, epoch, conf) 
            last_test_acc = self.test(net, epoch, conf)
        return last_train_acc, last_test_acc

def train(dataset, out_path=None, device='cpu', normalize_data=True, **kwargs):
    if 'cuda' in device: 
        assert torch.cuda.is_available()
    print("Using device: %s" % device)
    if dataset == 'mnist':
        ts = TrainingService_MNIST(home_dir='.', device=device)
    elif dataset == 'cifar-10':
        ts = TrainingService_CIFAR10(home_dir='.', device=device, 
                                     normalize_data=normalize_data)
    print('a', kwargs)
    cnn = ts.build_and_train(**kwargs)
    if out_path:
        torch.save(cnn, out_path)
        print('Model saved to %s' % out_path)


if __name__ == "__main__":
    import fire
    fire.Fire(train)