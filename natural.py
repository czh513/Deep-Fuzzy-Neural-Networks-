'''
Evaluate a model on natural examples
'''

from scipy.special import softmax
import torch
import torchvision
import numpy as np
import json
import sys

def compute_max_probs(preds):
    preds_softmax = softmax(preds, axis=1)
    max_probs = np.max(preds_softmax, axis=1)
    return max_probs

def evaluate(dataset=None, model_path=None):
    if dataset == 'mnist':
        test_dataset = torchvision.datasets.MNIST(
            root='./mnist', train=False, download=False,
            transform=torchvision.transforms.ToTensor(), 
        )
    elif dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(
            root='./cifar10', train=False, download=False,
            transform=torchvision.transforms.ToTensor(), 
        )
    else:
        raise ValueError('Unsupported dataset: ' + str(dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print('Evaluating %s on %s (clean)' % (model_path, dataset), file=sys.stderr)

    accuracies = []
    max_probs = []
    for x, y in test_loader:
        preds, _ = model(x)
        preds = preds.detach().numpy()
        accuracies.append((preds.argmax(axis=1) == y.numpy()).mean())
        max_probs.extend(compute_max_probs(preds).tolist())
    print(json.dumps({
        'model_path': model_path,
        'attack': 'none',
        'accuracies': accuracies,
        'max_probs': max_probs
    }))

if __name__ == "__main__":
    import fire
    fire.Fire(evaluate)