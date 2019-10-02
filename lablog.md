# Thu 3 Oct 2019

Trained a small CNN model on MNIST and extracted activations.

Looking at some samples, my hypothese of leaky formula and superimposition seem to be true.
Will need to do more quantitative evaluation...

TODO: also check what clustering and tree models come up with.

TODO: try to defend one neuron before applying to the whole network.

Question: should an ideal pattern be sparse actually? 25 features (5x5 filter) are already small.
Perhaps superimposed patterns can't be disentangled by selecting sets of features to look at
but by assigning patterns to "subneurons" and maintaining them during refining.
