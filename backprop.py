import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(z, 0)

def relu_prime(z):
    return (z > 0) * 1

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - np.pow(tanh(z), 2)

def cost(actual, expected):
    return 0.5 * (actual - expected) ** 2

def cost_prime(actual, expected):
    return actual - expected

class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.initialize_weights()
        self.caches = []

    def create_caches(self):
        # Shape: (self.layer_sizes[i],)
        # first layer always None
        preactivation_cache = [None for i in range(len(self.layer_sizes))]

        # Shape: (self.layer_sizes[i],)
        activation_cache = [None for i in range(len(self.layer_sizes))]

        # Shape: (self.layer_sizes[i],)
        # first layer always None
        error_cache = [None for i in range(len(self.layer_sizes))]

        # Shape: (self.layer_sizes[i], self.layer_sizes[i - 1])
        # first layer always None
        weight_grad_cache = [None for i in range(len(self.layer_sizes))]

        caches = (
            preactivation_cache,
            activation_cache,
            error_cache,
            weight_grad_cache
        )
        return caches

    def initialize_weights(self):
        self.weights = []
        self.biases = []
        for i in range(1, len(self.layer_sizes)):
            weights = np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1])
            bias = np.random.randn(self.layer_sizes[i])
            self.weights.append(weights)
            self.biases.append(bias)

    def propagate(self, input, activation=sigmoid, cache_activations=False):
        if input.shape != (self.layer_sizes[0],):
            raise Exception("Invalid input shape")
        if cache_activations:
            if not self.caches:
                raise Exception("Caches not created")
            self.caches[-1][1][0] = input

        current_layer = input.T
        for i in range(0, len(self.layer_sizes) - 1):
            weighted_inputs = np.dot(self.weights[i], current_layer) + self.biases[i]
            current_layer = activation(weighted_inputs)
            if cache_activations:
                self.caches[-1][0][i + 1] = weighted_inputs
                self.caches[-1][1][i + 1] = current_layer

        return current_layer.T

    def backpropagate(self,
                      inputs,
                      expected_outputs,
                      learning_rate=3,
                      cost_func=cost,
                      cost_func_grad=cost_prime,
                      activation=sigmoid,
                      activation_grad=sigmoid_prime):
        self.caches.clear()
        for input, exp_output in zip(inputs, expected_outputs):
            cache = self.create_caches()
            self.caches.append(cache)
            preactivation_cache = cache[0]
            activation_cache = cache[1]
            error_cache = cache[2]
            weight_grad_cache = cache[3]

            # Feedforward
            output = self.propagate(input, activation, cache_activations=True)

            # Output error
            last_layer_cost_grad = cost_func_grad(output, exp_output)
            last_layer_activation_grad = activation_grad(preactivation_cache[-1])
            last_layer_error = last_layer_cost_grad * last_layer_activation_grad
            error_cache[-1] = last_layer_error
            weight_grad_cache[-1] = np.outer(error_cache[-1], activation_cache[-2])

            # Backpropagation
            for i in range(len(self.layer_sizes) - 2, 0, -1):
                # iterate backwards until first layer
                backprop = np.dot(self.weights[i].T, error_cache[i + 1])
                error_cache[i] = backprop * activation_grad(preactivation_cache[i])
                weight_grad_cache[i] = np.outer(error_cache[i], activation_cache[i - 1])

        # Gradient Descent
        for i in range(1, len(self.layer_sizes)):
            weight_error_grads = [self.caches[j][3][i] for j in range(len(self.caches))]
            bias_error_grads = [self.caches[j][2][i] for j in range(len(self.caches))]
            avg_weight_error_grad = np.sum(weight_error_grads, axis=0) / len(self.caches)
            avg_bias_error_grad = np.sum(bias_error_grads, axis=0) / len(self.caches)
            self.weights[i - 1] -= learning_rate * avg_weight_error_grad
            self.biases[i - 1] -= learning_rate * avg_bias_error_grad

def main():
    network = Network([1, 32, 32, 1])
    inputs = np.linspace(-5, 5, 1000).reshape((1000, 1))
    outputs = np.sin(inputs) / inputs / 2 + 0.5

    import time
    start = time.perf_counter()
    for i in range(200):
        network.backpropagate(inputs, outputs)
    print("Finished training in", time.perf_counter() - start)

    x = []
    y = []
    for i in range(len(inputs)):
        output = network.propagate(inputs[i])
        x.append(inputs[i])
        y.append(output[0])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, c="r")
    ax1.scatter(inputs.reshape((1000,)), outputs.reshape((1000,)), c="b")
    plt.show()

main()
