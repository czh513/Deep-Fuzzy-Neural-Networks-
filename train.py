import os
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, mse_loss
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
from negaugment import *

def gaussian_noise(epsilon):
    transform_func = lambda x : (x + torch.randn_like(x)*epsilon)
    return torchvision.transforms.Lambda(transform_func)

def param_mean(vals):
    ''' This method takes one mean per layer (assuming each tensor comes from one layer) and then sum them together. 
    The rationale is that the scale of regularization signal shouldn't be dependent on the depth of the network. '''
    return sum(val.mean() for val in vals)
    # alternative formula
    # return sum((val.mean(dim=1) if val.ndim == 2 else val).sum() 
    #            for val in vals)

class WeightedMSELoss(object):

    def __init__(self, possitive_weight, negative_weight):
        self.possitive_weight = possitive_weight
        self.negative_weight = negative_weight

    def __call__(self, preds, labels):
        raw_loss = (preds-labels.float())**2
        unit_weights = torch.ones_like(raw_loss)
        weighted_loss = raw_loss * torch.where(labels == 1, 
                unit_weights*self.possitive_weight, 
                unit_weights*self.negative_weight)
        return weighted_loss.mean()

class TrainingService(object):

    def compute_loss(self, epoch, cnn, train_x, train_y, output, conf):
        orig_train_y = train_y
        if conf['use_mse']:
            train_y = one_hot(train_y, num_classes=self.num_classes).float()
            output = output.sigmoid()
            if conf['mse_weighted']:
                if conf['use_scrambling'] or conf['use_overlay']:
                    loss_func = WeightedMSELoss(19, 1) # hard code for now...
                else:
                    loss_func = WeightedMSELoss(9, 1) # hard code for now...
            else:
                loss_func = nn.MSELoss()
        else:
            loss_func = nn.CrossEntropyLoss()
        main_loss = loss_func(output, train_y)

        neg_training_loss = torch.tensor(0.0).to(self.device)
        if (conf['use_scrambling'] or conf['use_overlay']) and epoch >= 1:
            assert conf['use_mse'], "Softmax networks can't handle all-negative input"
            f = self.scramble if conf['use_scrambling'] else OverlayNegativeSamples()
            neg_x = f(train_x, orig_train_y).to(self.device)
            neg_y = torch.zeros(neg_x.shape[0], self.num_classes).to(self.device)
            neg_output, _ = cnn(neg_x)
            neg_training_loss += loss_func(neg_output, neg_y)

        reg_loss = torch.tensor(0.0).to(self.device)
        if epoch >= conf['regularization_start_epoch']:
            if conf['regularization'] in ('max-fit', 'max-margin'):
                assert (conf['l1'] > 0) or (conf['l2'] > 0), "Strength of regularization must be specified"
                if conf['l1'] > 0:
                    reg_loss += conf['l1'] * param_mean(w.abs() for w in cnn.weights)
                if conf['l2'] > 0:
                    reg_loss += conf['l2'] * param_mean(w*w for w in cnn.weights)
            if conf['regularization'] == 'max-fit':
                assert (conf['bias_l1'] > 0), \
                    "For max-fit, strength of bias regularization must be specified"
                reg_loss += conf['bias_l1'] * param_mean(b for b in cnn.bias) # notice: no abs()

        loss = main_loss + reg_loss + neg_training_loss
        return loss, main_loss, reg_loss, neg_training_loss

def print_control_params(net):
    # tried measuring all weights and quadratic weights here but mean and std
    # weren't helpful. whatever changes they have they don't reflect on overall stats
    print('  Curvature multiplier: %.4f, log strength %.4f'
        %(models.extract_curvature_strength(net),
            models.extract_log_strength(net)))


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
        config_defaults = {
            'use_mse': False, 'lr': 0.001, 'out_path': None, 'train_batch_size': 64, 
            'regularization': None, 'regularization_start_epoch': 2, 'l1': 0, 'l2': 0, 
            'bias_l1': 0, 'use_scrambling': False, 'use_overlay': False,
            'use_elliptical': False, 'use_quadratic': False, 'mse_weighted': False,
        }

        model_kwargs = {k:v for k, v in kwargs.items() if k in models.CNN.config_defaults}
        cnn = models.CNN(**model_kwargs)
        print(cnn)  # net architecture

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
                loss, _, _, _ = self.compute_loss(epoch, cnn, train_x, train_y, output, conf)
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

cifar_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

class TrainingService_CIFAR10(TrainingService):

    def __init__(self, home_dir, device='cpu', gaussian_noise_epsilon=0.5,
                 num_batches_per_epoch=-1):
        self.home_dir = home_dir
        self.prepare_data(gaussian_noise_epsilon)
        self.device = device
        self.num_batches_per_epoch = num_batches_per_epoch # set to a small value for testing
        self.scramble = ChoiceScramble([PixelScramble(), BlockScramble(3), BlockScramble(9)])
        self.num_classes = 10
        self.last_train_acc = self.last_test_acc = None

    def prepare_data(self, gaussian_noise_epsilon):
        # Data
        print('==> Preparing data..')
        # Tried random rotating, gaussian noise, and random erasing as in MNIST 
        # but models fail to learn. Didn't try them individually.

        transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomRotation(degrees=10),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*cifar_stats),
                # it's important to apply noise _after_ normalization
                gaussian_noise(gaussian_noise_epsilon),
        ])
        transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*cifar_stats)
        ])
        self.trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train)
        self.testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform_test)

    # Training
    def train(self, net, epoch, conf):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = total = 0
        batch_idx = 0
        for _ in range(conf['batch_size_multiplier']):
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs, _ = net(inputs)
                loss, main_loss, reg_loss, _ = self.compute_loss(epoch, net, inputs, targets, outputs, conf)
                loss.backward()
                self.optimizer.step()

                train_loss += main_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = correct/total
                if batch_idx % conf['report_interval'] == 0:
                    print(batch_idx, '/', len(self.trainloader)*conf['batch_size_multiplier'], 
                        ': Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*acc, correct, total))
                    print('  Regularization loss: %f' % reg_loss.item())
                    print_control_params(net)
                if self.num_batches_per_epoch > 0 and batch_idx >= self.num_batches_per_epoch:
                    break # end it early so that we can test the code
                batch_idx += 1
        return acc

    def train_acc_estimate(self, net, epoch, conf):
        ''' Use this to detect training collapse '''
        net.eval()
        correct = total = 0
        for inputs, targets in itertools.islice(self.trainloader, 25):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs, _ = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct/total

    def test(self, net, epoch, conf):
        net.eval()
        test_loss = 0
        correct = total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _ = net(inputs)

                if conf['use_mse']:
                    outputs = outputs.sigmoid()
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

    def build_and_train(self, **kwargs):
        config_defaults = {
            'use_mse': False, 'lr': 0.01, 'out_path': None, 'train_batch_size': 128, 
            'regularization': None, 'regularization_start_epoch': 10, 'l1': 0, 'l2': 0,
            'bias_l1': 0, 'use_scrambling': False, 'use_overlay': False,
            'use_elliptical': False, 'use_quadratic': False, 
            'log_strength_inc': 0.001, 'log_strength_start': 0.001, 'log_strength_stop': 1,
            'batch_size_multiplier': 1, 'report_interval': 150,
            'curvature_multiplier_inc': 1e-4, 'curvature_multiplier_start': 0,
            'curvature_multiplier_stop': 1, 'n_epochs': 40, 'mse_weighted': False,
        }
        unrecognized_params = [k for k in kwargs
                               if not (k in models.VGG.config_defaults or k in config_defaults)]
        assert not unrecognized_params, 'Unrecognized parameter: ' + str(unrecognized_params) 

        conf = {**config_defaults, **kwargs}
        print('Using training service config:', conf)
        models.log_strength_inc = float(conf['log_strength_inc'])
        models.log_strength_start = float(conf['log_strength_start'])
        models.log_strength_stop = float(conf['log_strength_stop'])
        models.curvature_multiplier_inc = float(conf['curvature_multiplier_inc'])
        models.curvature_multiplier_start = float(conf['curvature_multiplier_start'])
        models.curvature_multiplier_stop = float(conf['curvature_multiplier_stop'])

        model_kwargs = {k:v for k, v in kwargs.items() if k in models.VGG.config_defaults}
        net = models.VGG(**model_kwargs)
        print(net)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, shuffle=True, num_workers=2, 
                batch_size=conf['train_batch_size']*conf['batch_size_multiplier'])
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=256, shuffle=False, num_workers=2)
        # tried ADAM already: it works for ReLU but fail to train ReLog (it doesn't just overfit,
        # it increases the loss after a few epochs)
        net = net.to(self.device)
        for epoch in range(conf['n_epochs']):
            self.optimizer = optim.SGD(net.parameters(), lr=conf['lr'], momentum=0.9)
            self.train(net, epoch, conf)
            new_train_acc = self.train_acc_estimate(net, epoch, conf)
            training_has_collapsed = (self.last_train_acc and new_train_acc < 0.5 * self.last_train_acc)
            if training_has_collapsed:
                if conf['regularization'] and epoch >= conf['regularization_start_epoch']:
                    print("Training might have collapsed because of excessive regularization, "
                          "please adjust hyperparams, aborting...")
                    if conf['out_path'] and os.path.exists(conf['out_path']):
                        net = torch.load(conf['out_path']) # recover
                    break
                else: # attempt recovery
                    if conf['out_path'] and os.path.exists(conf['out_path']):
                        models.freeze_hyperparams = True
                        net = torch.load(conf['out_path']) # recover
                        for _ in range(5):
                            print('\n=== Trying to overcome collapse point ===')
                            self.train(net, epoch, conf)
                        print('\nNow, retrying normal training')
                        models.freeze_hyperparams = False
                    else:
                        print("Collapse of training detected but no model available on disk")
            else: # only test and write model in normal state
                self.last_train_acc = new_train_acc
                self.last_test_acc = self.test(net, epoch, conf)
                if conf['out_path']:
                    torch.save(net, conf['out_path'])
                    print('Model saved to %s' % conf['out_path'])
        return net

def train(dataset, device='cpu', **kwargs):
    if 'cuda' in device: 
        assert torch.cuda.is_available()
    print("Using device: %s" % device)
    if dataset == 'mnist':
        ts = TrainingService_MNIST(home_dir='.', device=device)
    elif dataset == 'cifar-10':
        ts = TrainingService_CIFAR10(home_dir='.', device=device)
    else:
        raise ValueError("Unsupported dataset: " + str(dataset))
    cnn = ts.build_and_train(**kwargs)
    out_path = kwargs.get('out_path')
    if out_path:
        torch.save(cnn, out_path)
        print('Model saved to %s' % out_path)


if __name__ == "__main__":
    import fire
    fire.Fire(train)