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

    def initialize_weights(self):
        self.weights = []
        self.biases = []
        for i in range(1, len(self.layer_sizes)):
            weights = np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1])
            bias = np.random.randn(self.layer_sizes[i])
            self.weights.append(weights)
            self.biases.append(bias)

    def propagate(self, input, activation=sigmoid):
        current_layer = input.T
        for i in range(0, len(self.layer_sizes) - 1):
            biases = self.biases[i].reshape((self.biases[i].shape[0], 1))
            weighted_inputs = np.dot(self.weights[i], current_layer) + biases
            current_layer = activation(weighted_inputs)

        return current_layer.T

    def backpropagate(self,
                      inputs,
                      expected_outputs,
                      learning_rate=3,
                      cost_func=cost,
                      cost_func_grad=cost_prime,
                      activation=sigmoid,
                      activation_grad=sigmoid_prime):
        preactivation_cache = [None for i in range(len(self.layer_sizes))]
        activation_cache = [None for i in range(len(self.layer_sizes))]
        error_cache = [None for i in range(len(self.layer_sizes))]
        weight_grad_cache = [None for i in range(len(self.layer_sizes))]

        # Feedforward
        current_layer = inputs.T
        activation_cache[0] = current_layer
        for i in range(0, len(self.layer_sizes) - 1):
            biases = self.biases[i].reshape((self.biases[i].shape[0], 1))
            weighted_inputs = np.dot(self.weights[i], current_layer) + biases
            current_layer = activation(weighted_inputs)
            preactivation_cache[i + 1] = weighted_inputs
            activation_cache[i + 1] = current_layer

        network_output = current_layer

        # Output error
        last_layer_cost_grad = cost_func_grad(network_output, expected_outputs.T)
        last_layer_activation_grad = activation_grad(preactivation_cache[-1])
        last_layer_error = last_layer_cost_grad * last_layer_activation_grad
        error_cache[-1] = last_layer_error
        weight_grad_cache[-1] = np.dot(error_cache[-1], activation_cache[-2].T)

        # Backpropagation
        for i in range(len(self.layer_sizes) - 2, 0, -1):
            # iterate backwards until first layer
            backprop = np.dot(self.weights[i].T, error_cache[i + 1])
            error_cache[i] = backprop * activation_grad(preactivation_cache[i])
            weight_grad_cache[i] = np.dot(error_cache[i], activation_cache[i - 1].T)

        # Gradient Descent
        for i in range(1, len(self.layer_sizes)):
            avg_weight_error_grad = weight_grad_cache[i] / len(inputs)
            avg_bias_error_grad = np.sum(error_cache[i], axis=1) / len(inputs)
            self.weights[i - 1] -= learning_rate * avg_weight_error_grad
            self.biases[i - 1] -= learning_rate * avg_bias_error_grad

def main():
    network = Network([1, 16, 16, 1])
    inputs = np.linspace(-5, 5, 1000).reshape((1000, 1))
    outputs = np.sin(inputs) / inputs / 2 + 0.5

    import time
    start = time.perf_counter()
    for i in range(200):
        network.backpropagate(inputs, outputs)
    print("Finished training in", time.perf_counter() - start)

    network_output = network.propagate(inputs)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(inputs.reshape((1000,)), network_output.reshape((1000,)), c="r")
    ax1.scatter(inputs.reshape((1000,)), outputs.reshape((1000,)), c="b")
    plt.show()

if __name__ == "__main__":
    main()
