# Neural Networks

- [perceptron.py](perceptron.py): basic adder using perceptrons (heaviside step function)
- [perceptron_pure_matrix.py](perceptron_pure_matrix.py): same as above, but using a larger matrix to include the bias translation in the multiplication
- [backprop.py](backprop.py): backpropagation algorithm with stochastic gradient descent
- [backprop_matrix.py](backprop_matrix.py): same as above, but allowing propagation of multiple inputs at once, also cross-entropy cost function
- [backprop_batch.py](backprop_batch.py): same as above, but using mini batches for gradient descent
- [backprop_regularized.py](backprop_regularized.py): same as above, but with l2 regularization, also reducing starting weights based on input layer size
- [backprop_momentum.py](backprop_momentum.py): same as above, but sutskever nesterov momentum-based
- [backprop_activations.py](backprop_activations.py): same as above, but with more activation functions and per-layer activation functions
- [backprop_initializations.py](backprop_initializations.py): same as above, but with per-layer weight intialization depending on activation
- [mnist.py](mnist.py): testing the neural network with the MNIST dataset

sutskever-nesterov acceleration momentum-based mini-batch stochastic gradient descent feed-forward neural network with l2 regularization and weight initialization
