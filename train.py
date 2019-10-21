import os
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from models import CNN, VGG, Spherical, inc_curvature_multiplier, reset_curvature_multiplier
from utils import grouper_variable_length
from time import time
import torch.backends.cudnn as cudnn
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

    def __init__(self, home_dir, epsilon=0.3, batch_size=64, n_epochs=2, timeout_sec=120, 
                 test_freq=100, lr=0.001, num_classes=10): 
        self.home_dir = home_dir
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.timeout_sec = timeout_sec
        self.test_freq = test_freq
        self.lr = lr
        self.epsilon = epsilon
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using device: %s" % self.device)
        self.load_data()
        self.cnn_func = CNN
        self.num_classes = num_classes
        self.scramble = ChoiceScramble([PixelScramble(), BlockScramble(5), BlockScramble(7)])

    def load_data(self):
        self.data_dir = os.path.join(self.home_dir, 'mnist')
        download_data = not(os.path.exists(self.data_dir)) or not os.listdir(self.data_dir)
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 1.0)),
            torchvision.transforms.RandomRotation(degrees=10),
            torchvision.transforms.ToTensor(),
            gaussian_noise(self.epsilon),
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

    def build_and_train_cnn(self, name=None, regularization=None, l1=0, l2=0, 
                            bias_l1=0, bias_l2=0, use_scrambling=False, **kwargs):
        use_sigmoid_out = kwargs.get('use_sigmoid_out', False)
        self.use_spherical = kwargs.get('use_spherical', False)
        cnn = self.cnn_func(**kwargs)
        print(cnn)  # net architecture
        self.train_loop_with_timeout(cnn, use_sigmoid_out, regularization, 
                                     l1, l2, bias_l1, bias_l2, use_scrambling)
        if name:
            out_path = os.path.join(self.home_dir, 'output', '%s.pkl' % name)
            torch.save(cnn, out_path)
            print('Model saved to %s' % out_path)
        return cnn

    def train_loop_with_timeout(self, cnn, use_sigmoid_out=False, regularization=None, 
                                l1=0, l2=0, bias_l1=0, bias_l2=0, use_scrambling=False):
        train_loader = Data.DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset=self.test_data, batch_size=2000, shuffle=True)
        optimizer = torch.optim.Adam(cnn.parameters(), amsgrad=True, lr=self.lr)
        loss_func = nn.MSELoss() if use_sigmoid_out else nn.CrossEntropyLoss()

        cnn.to(self.device)
        started_sec = time()
        reset_curvature_multiplier()
        for epoch in range(self.n_epochs):
            batch_groups = grouper_variable_length(train_loader, self.test_freq)
            for batch_group, (test_x, test_y) in zip(batch_groups, test_loader): 
                for (train_x, train_y) in batch_group:
                    cnn.train()
                    train_x = train_x.to(self.device)
                    train_y = train_y.to(self.device)
                    output, _ = cnn(train_x)
                    if use_sigmoid_out:
                        train_y = one_hot(train_y, num_classes=self.num_classes).float()
                    loss = loss_func(output, train_y)
                    if use_scrambling and epoch >= 1:
                        assert use_sigmoid_out, "Softmax networks can't handle scrambled input"
                        scramble_x = self.scramble(train_x)[:train_x.shape[0]]
                        scramble_y = torch.zeros(scramble_x.shape[0], self.num_classes).to(self.device)
                        scramble_output, _ = cnn(scramble_x)
                        loss += loss_func(scramble_output, scramble_y)
                    if regularization in ('max-fit', 'max-margin') and not self.use_spherical:
                        assert (l1 > 0) or (l2 > 0), "Strength of regularization must be specified"
                        if l1 > 0:
                            loss += l1 * tmean(w.abs() for w in cnn.weights)
                        if l2 > 0:
                            loss += l2 * tmean(w*w for w in cnn.weights)
                    if regularization == 'max-margin' and self.use_spherical:
                        assert (bias_l1 > 0) or (bias_l2 > 0), "For max-margin with spherical, strength of bias regularization must be specified"
                        if bias_l1 > 0:
                            loss += -bias_l1 * tmean(b for b in cnn.bias) # notice: no abs()
                        if bias_l2 > 0:
                            loss += -bias_l2 * tmean(b*b.abs() for b in cnn.bias) # notice: signed
                    if regularization == 'max-fit':
                        assert (bias_l1 > 0) or (bias_l2 > 0), "For max-fit, strength of bias regularization must be specified"
                        if bias_l1 > 0:
                            loss += bias_l1 * tmean(b for b in cnn.bias) # notice: no abs()
                        if bias_l2 > 0:
                            loss += bias_l2 * tmean(b*b.abs() for b in cnn.bias) # notice: signed
                    optimizer.zero_grad()           # clear gradients for this training step
                    loss.backward()                 # backpropagation, compute gradients
                    optimizer.step()                # apply gradients
                    if self.use_spherical:
                        inc_curvature_multiplier()
                cnn.eval()
                with torch.no_grad():
                    test_output, _ = cnn(test_x.to(self.device))
                    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
                    accuracy = float((pred_y == test_y.data.numpy()).astype(int).mean())
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy)
                
                elapsed_sec = time() - started_sec
                if elapsed_sec > self.timeout_sec:
                    print('Timeout (%.2f sec), training is terminated' % elapsed_sec)
                    return

class CIFAR10_TrainingService(object):

    def __init__(self, home_dir, normalize_data):
        self.home_dir = home_dir
        self.prepare_data(normalize_data)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device: %s" % self.device)

    def prepare_data(self, normalize_data):
        # Data
        print('==> Preparing data..')
        cifar_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
        ] + ([torchvision.transforms.Normalize(*cifar_stats)] if normalize_data else []))
        transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
        ] + ([torchvision.transforms.Normalize(*cifar_stats)] if normalize_data else []))
        trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Training
    def train(self, net, epoch, out_path):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = net(inputs)
            if self.use_sigmoid_out:
                loss_targets = one_hot(targets, num_classes=10).float()
            else:
                loss_targets = targets
            loss = self.criterion(outputs, loss_targets)
            if self.strictening is not None and self.strictening > 0:
                weight_reg = tmean(w.abs() for w in net.weights)
                bias_reg = tmean(b for b in net.bias) # notice: no abs()
                loss += self.strictening * (weight_reg + bias_reg)
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 5)
            self.optimizer.step()
            if self.use_spherical:
                inc_curvature_multiplier()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 150 == 0:
                print(batch_idx, '/', len(self.trainloader), ': Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if out_path:
            torch.save(net, out_path)
            print('Model saved to %s' % out_path)

    def test(self, net, epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = net(inputs)

                if self.use_sigmoid_out:
                    loss_targets = one_hot(targets, num_classes=10).float()
                else:
                    loss_targets = targets
                loss = self.criterion(outputs, loss_targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print('Test eval: Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def build_and_train(self, name=None, n_epochs=20, strictening=None, **kwargs):
        self.vgg_name = kwargs.get('vgg_name', 'VGG11')
        self.use_sigmoid_out = kwargs.get('use_sigmoid_out', False)
        self.use_relog = kwargs.get('use_relog', False)
        self.use_spherical = kwargs.get('use_spherical', False)
        self.strictening = strictening
        net = VGG(**kwargs)
        print(net)
        net = net.to(self.device)
        out_path = os.path.join(self.home_dir, 'output', '%s.pkl' % name) if name else None
        self.optimizer = optim.Adam(net.parameters(), lr=0.1)
        self.criterion = nn.MSELoss() if self.use_sigmoid_out else nn.CrossEntropyLoss()
        reset_curvature_multiplier()
        for epoch in range(n_epochs):
            # since it takes a looong time to train, we'll save every epoch
            self.train(net, epoch, out_path) 
            self.test(net, epoch)

if __name__ == "__main__":
    # test the code
    # ts = CIFAR10_TrainingService(home_dir='.', normalize_data=False)
    # ts.build_and_train(vgg_name='VGG11', n_epochs=1, use_maxout='max', use_relog=True, use_spherical=True)
    ts = TrainingService(home_dir='.', n_epochs=1)
    ts.build_and_train_cnn(use_scrambling=True, use_maxout='min', min_folding_factor=4, use_relog=True, 
                           use_elliptical=True, use_sigmoid_out=True, regularization='max-margin', l1=0.1, bias_l1=0.1)    