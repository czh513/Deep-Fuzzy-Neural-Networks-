

# Reproducing results

Notice that the following steps only work on the computing environment they 
were created for (which is the Dutch cluster Cartesius). Please read the 
scripts and adapt them to your computing environment.

## MNIST experiments

1. Train models: `drake %mnist-models`
2. Evaluate models: `drake %mnist-results`

## CIFAR-10 experiments

1. Train models: `drake %cifar10-models`
2. Evaluate models: `drake %cifar10-results`

## Figures and tables in the paper

1. ReLog activation function figure: `reporting/activation-functions.ipynb`
2. Pattern fitting figure (two-dot problem): `reporting/two-dots.ipynb`
3. Regularization figure: `reporting/fitting.ipynb`
4. Negative examples figure: `reporting/overlay.ipynb`
5. Activation on noise: `reporting/distributed-vs-local.ipynb`
6. MNIST and CIFAR-10 result tables: `reporting/results-table.ipynb`
