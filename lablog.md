# Thu 3 Oct 2019

Trained a small CNN model on MNIST and extracted activations.

Looking at some samples, my hypothese of leaky formula seem to be true because 50% of the times,
the neuron has a weak activation of ~6% max activation.
Will need to do more quantitative evaluation...

Question: should an ideal pattern be sparse actually? 25 features (5x5 filter) are already small.
Perhaps superimposed patterns can't be disentangled by selecting sets of features to look at
but by assigning patterns to "subneurons" and maintaining them during refining.

Visualized patterns that activate my target neuron. Looks like there are several distinct patterns indeed.
One is a diagonal ridge, one is top-left corner, another for top horizontal edge and there are a few
cases of top-right corner as well.

Comparing the weights and patterns that end up activating the neuron the most (top 1%), we can see that
the weights are the average of a few distinct patterns:

- Weights: ![weights](images/conv1-neuron-no7-weights.png)
- Pattern 1: ![pattern 1](images/conv1-neuron-no7-pattern1.png)
- Pattern 2: ![pattern 2](images/conv1-neuron-no7-pattern2.png)
- Pattern 3: ![pattern 3](images/conv1-neuron-no7-pattern3.png)

Apparently, when it comes to weights, you need the _oposite_ of sparsity.
A weight of zero means uncertainty and leaves wiggle room for attackers.
You need a way to encode multiple patterns with high certainty.

Other neurons show similar pattern of weights with a lot of uncertainty
and can't be mapped directly to a clear pattern of input. The following 
plot shows neuron weights scaled (but not translated) to fit within \[-1, 1\].
There are many black cells which stand for zero weights and all subplots
look like a patchwork without any discernible pattern.

![neurons](images/conv1-neurons.png)

A denser and more distinguishing neuron might result in _sparser_ activations
in the sense that it will return absolute zeros more often.

Given this analysis, tree models are not a good approach because they tend to 
encourage sparseness. We could throw away the idea of "disentangling" a network
into a better one but go directly to an architecture that encourages the
distintion and disentanglement of patterns.

# Sat 5 Oct 2019

Playing around with rectified log and maxout.

An immediate benefit of rectified log vs rectified linear is that the new activation function
creates nets that are less sensitive to initialization. Following 
[this blog post](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79), 
I tested how many layers one can propagate before the output explodes. ReLU get past 28 while
ReLog went through all 100 layers. Obviously, it can't solve the problem of too small activations,
it doesn't touch the left part of the plot.

Maxout seems to be effective in tearing out patterns but so is adding more neurons into the
_previous_ layer. This is based on my eye-balling, I'll need a quantitative way to measure it.

# Sun 6 Oct 2019

Fortified the training procedure with transformation of the input images: crop-resize, rotate,
random erasure, Gaussian noise. This should strengthen the baseline against adversarial examples
so that the comparison is valid.

Implemented another variant: sigmoid output with MSE loss. Because I'm testing the extent
neural nets resemble a set of logical propositions, each class should be a proposition
independent of the output of other classes. I also believe this loss is stricter than than
cross-entropy with softmax output and should help with fending off attacks.


TODO: defend the model further by generating counter-examples that don't belong to any of the
given classes (open-world setting).

# Wed 9 Oct 2019

Figured out a way to make relog more similar to sigmoid: changing it from:

    y = log10(relu(x)+1)
    
into:
    
    y = log10(relu(x)+0.01)+2
    
The results on adversarial examples increased to >60%, without other tricks :-O
This is huge... I will definitely write a paper!

Brought notebooks to Google colab to scale up experiments using GPUs. 
Things that took 20 min now finish in 45 sec! Having trouble 
with importing python files though.

Adjusted the relog formula a bit:

    y = log100(relu(x)+0.01)+1

The result with all the tricks except "strictening" is 90%. On adversarial
attack (fast gradient). Insane! Did I make a mistake somewhere?

Attempted my hand with [JSMA](jsma.ipynb) but got some TensorFlow/PyTorch
mismatch problem.

Had a big dose of literature. Added 15+ papers to "adversarial-examples"
folder in my Mendeley library.

C&W is a much stronger attack. My best performance is with 
`cnn-mnist-relog-minmaxout_4_2-sigmoid-out-strictening_0.1`
which is still fooled for 50% cases. Interesting that "strictening"
improves the robustness against C&W whereas it didn't have much
effect on FGSM. 

Is "tightening" a better name? Or logic regularization?

# Fri 11 Oct

Researching CIFAR-10. This is the first time that I read about it
in detail.

# Sat 12 Oct

Used a day to find [a good implementation](https://github.com/kuangliu/pytorch-cifar)
and adapt it to my need. Preliminary results seem to support my story
but it's running too slowly and Google Colab keeps breaking connection
to data storage.

Running [this notebook](https://colab.research.google.com/drive/1Kh0sTuEHXqNjGhEMBrOIHlxWK3UUWFRw#scrollTo=WKJDICIZlb9K&uniqifier=3)
to train some CIFAR models and using
[this one](https://colab.research.google.com/drive/1BXsR0s524p9lLeAvXwAAwgNm0SgJ4hOp#scrollTo=XH6gaV34Na-M)
to evaluate them against attacks.

State-of-the-art for CIFAR-10: https://paperswithcode.com/paper/adversarial-defense-by-restricting-the-hidden

TODO: polishing the C&W attack code
TODO: run one transfer attack and finish there

TODO: compare my "strictening" L1 and L2 with normal L1 and L2

# Sun 13 Oct

<s>Found out</s> why I got low results for ReLog on CIFAR: the initialization is bad so 
my models weren't learning. Spent an afternoon working on a new initialization formula
but didn't get any luck...

Found out why ReLog didn't work with CIFAR: negative numbers. 
With ReLog, positive weights encode positive atoms (found --> add score)
and negative ones encode negative atoms (found --> minus score).
But because of normalization to Gaussian distribution, the absence of a color
(which is used to be zero) becomes a strongly negative number. Meeting with this negative
number leads to a decrease in the final score whereas it should have had no effect.
Middle-strong color which should have mild effect now has no effect so the relationship
is all messed up. It has nothing to do with initialization (although a custom-made one might
perform better)!

Not sure how ReLU performs alright (even better) with negative numbers...

Everything seems to be working now, except that I can't wait long enough to see how 
models perform. It's time to get some GPU...?

# Mon 14 Oct

Made this image to illustrate the advantage of min/max (and/or):

![](images/relu-min-max.png)

# Tue 15 Oct

Realized that I could apply the kernel trick to filters to improve
the fit to eliptical point clouds. It does decrease the success rate
of C&W attack on MNIST from 58% (model=`cnn-mnist-relog-minmaxout_4_2-sigmoid-out-strictening_1.pkl`)
to 51% (model=`cnn-mnist-relog-kernel-minmaxout_4_2-sigmoid-out.pkl`).
However, two problems arise:

1. the regularization formula for linear model doesn't apply any more,
using it will actually make models _less_ robust
2. the loss blows up to infinity. Replacing Kaiming with dynamic 
initialization helps with the first epoch but the problem persists. 
Gradient clipping doesn't help. Perhaps this problem is related to 
the first problem...

ReLog helps with reigning in the loss, after adding it, the loss doesn't
explode any more but the deep model still struggle to learn...

# Sat 19 Oct

No, actually a quadratic kernel didn't have any negative effect. I got new
best result with it:

    Evaluating model: /content/gdrive/My Drive/Colab Notebooks/newlogic/output/cnn-mnist-relog-kernel-minmaxout_4_2-sigmoid_out-max_margin.pkl

    [INFO 2019-10-19 13:59:40,200 cleverhans] Constructing new graph for attack CarliniWagnerL2

    Accuracy: 0.560

I must have implemented something incorrectly before. Just to be sure...
the current commit is: 8433041fbf5eee2ab294b320faff2f81f6bfe833
Important notebooks:

- [training CNN on MNIST](notebooks/mnist_train.ipynb)
- [evaluating using FGM](notebooks/mnist_fgm.ipynb)
- [evaluating using C&W](notebooks/mnist_cw.ipynb)

Struggling to measure C&W accuracy on the full MNIST test set, it runs so
slowly...

# Sun 20 Oct

Looked at some examples of adversarial images generated by C&W and many of them make a lot of sense.
Here are some cherry-picked examples:

![example C&W](images/cw-examples.png)
![example C&W](images/cw-examples2.png)
![example C&W](images/cw-examples3.png)

I think the not-so-strong performance is due to 
a lack of hyperparameter tuning instead of a fundamental problem.
Another thing I notice is although the (sigmoid) output is over-confident, if we
normalize them using softmax, we'll get close to 50% in all cases.

Got new results, every still makes sense. Got blocked from Google Colab
for running long-term jobs :)) Going to move to Cartesius anyway.

# Mon 21 Oct

If I report confidence, it needs to be under this attack:
https://github.com/tensorflow/cleverhans/blob/688fe64de5bda82895cc8729348a5d761c5e7813/cleverhans/attacks/max_confidence.py

In essence, attacks only work on the rate of fooling models but don't 
optimize the confidence on the fooled prediction so it's been known
before that we could "defend" models by confidence thresholding and
there was already a method to circumvent this defense.

Talked to Antske, agreed to cut it into a short paper and wrap it up ASAP.

Should I discuss the overfitting hypothesis of adversarial examples?
It was already discussed in Warde-farley and Goodfellow (2018), 
section 1.2.3 but maybe I could provide supporting evidence?

Warde-farley, D., & Goodfellow, I. (2018). Adversarial Perturbations of Deep Neural Networks. In Perturbations, Optimization, and Statistics. https://doi.org/10.7551/mitpress/10761.003.0012

Generalized [spherical units](notebooks/spherical_filters.ipynb) into
[elliptical units](notebooks/elliptical_filters.ipynb) such that 
it is possible to ignore certain input variable and the regularization
formulae become more natural.

Moving to Cartesius... the `environment.yaml` file wasn't useable 
(because... of course :-|) so I had to install packages manually.

# Tue 22 Oct

Training a few baseline model (VGG11 and VGG13) on Cartesius so that
I know what kind of performance can be expected in how much time.

    commit: 85ec98929bc2b2f9540765c850965f67b59b9537
    
    [minhle@int2 newlogic]$ sbatch scripts/train.job
    Submitted batch job 7008627

    results:
    tail -f output/train-vgg11.log
    tail -f output/train-vgg13.log

Initially, models fail to train. After much debugging, I found out that
it was due to an untested change I put it a while ago (gradient clipping 
to L_inf=5).

TODO: compare neurons of ReLU and logic nets to see if the latter is
"more logic" (what would that mean...?)

TODO: evaluate elliptical

TODO: measure average confidence on **only perturbed** images **under max-confidence attack**.

TODO: blackbox attack: https://github.com/tensorflow/cleverhans/blob/688fe64de5bda82895cc8729348a5d761c5e7813/tests_tf/test_mnist_blackbox.py