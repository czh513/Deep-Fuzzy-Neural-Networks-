'''
Approaches for negative sample augmentation.
'''
import torch
import numpy as np
import math

class PixelScramble(object):

    def __call__(self, input, y=None):
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
      
    def __call__(self, input, y=None):
        ''' Scramble pixels in a picture, without crossing color channels '''
        output = (input
                  [:,:,self.blocked_randperm(input.shape[2]),:]
                  [:,:,:,self.blocked_randperm(input.shape[3])])
        return output

class ChoiceScramble(object):

    def __init__(self, scrambles):
        self.scrambles = scrambles

    def __call__(self, input, y=None):
        i = np.random.choice(len(self.scrambles))
        return self.scrambles[i](input)

class OverlayNegativeSamples(object):

    def __call__(self, x, y):
        idx = torch.randint(x.shape[0], (x.shape[0], 2))
        mask = (y[idx[:,0]] != y[idx[:,1]])
        selected_x = x[idx[mask]]
        weights = (torch.rand(selected_x.shape[0], 2) * 0.3 + 0.7).to(x.device)
        scaled = selected_x * weights.reshape(selected_x.shape[0], 2, 1, 1, 1)
        overlay, _ = scaled.max(axis=1)
        assert overlay.shape[1:] == x.shape[1:]
        return overlay