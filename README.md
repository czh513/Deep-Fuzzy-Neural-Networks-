

# Reproducing results

Notice that the following steps only work on the computing environment they 
were created for (which is the Dutch cluster Cartesius). Please read the 
scripts and adapt them to your computing environment.

## MNIST experiments

0. Code version: `192f3d5118f82982c7ae6e2561b6a4a1726a205a`
1. Train models: `drake %mnist-models`
2. Wait until all jobs finish and run evaluation: `drake %mnist-results`
3. Continue at "Figures and tables" section

## CIFAR-10 experiments

0. Code version: `3a5c7896ffa925f733ba107188e9df08b3b5fc6d` (later versions will work differently)
1. Train models: `drake %cifar10-models`
2. Wait until all jobs finish and run evaluation: `drake %cifar10-results`
3. Continue at "Figures and tables" section

## Numbers, figures and tables in the paper

1. ReLog activation function figure: `reporting/activation-functions.ipynb`
2. Pattern fitting figure (two-dot problem): `reporting/two-dots.ipynb`
3. Regularization figure: `reporting/fitting.ipynb`
4. Negative examples figure: `reporting/overlay.ipynb`
5. Activation on noise: `reporting/distributed-vs-local.ipynb`
6. MNIST and CIFAR-10 result tables: `reporting/results-table.ipynb`
7. Curvature statistics in Section 4.1: `reporting/curvature.ipynb`