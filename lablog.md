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



