import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

    def __call__(self, z):
        return self.func(z)

    def deriv(self, z):
        return self.derivative(z)

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def _sigmoid_deriv(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s)

sigmoid = ActivationFunction(_sigmoid, _sigmoid_deriv)

def _relu(z):
    return np.maximum(z, 0)

def _relu_deriv(z):
    return (z > 0) * 1

relu = ActivationFunction(_relu, _relu_deriv)

def _tanh(z):
    return np.tanh(z)

def _tanh_deriv(z):
    return 1 - np.pow(np.tanh(z), 2)

tanh = ActivationFunction(_tanh, _tanh_deriv)

class CostFunction:
    def __init__(self, func, error):
        self.func = func
        self.error = error

    def __call__(self, actual, expected):
        return self.func(actual, expected)

    def error(self, actual, expected):
        return self.error(actual, expected)

def _quadratic(actual, expected):
    return 0.5 * (actual - expected) ** 2

def _quadratic_error(weighted, actual, expected):
    return (actual - expected) * _sigmoid_deriv(weighted)

quadratic = CostFunction(_quadratic, _quadratic_error)

def _cross_entropy(actual, expected):
    return -(expected * np.log(actual) + (1 - expected) * np.log(1 - actual))

def _cross_entropy_error(weighted, actual, expected):
    return actual - expected

cross_entropy = CostFunction(_cross_entropy, _cross_entropy_error)

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
                      cost_func=cross_entropy,
                      activation=sigmoid):
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
        last_layer_error = cost_func.error(
            preactivation_cache[-1], network_output, expected_outputs.T)
        error_cache[-1] = last_layer_error
        weight_grad_cache[-1] = np.dot(error_cache[-1], activation_cache[-2].T)

        # Backpropagation
        for i in range(len(self.layer_sizes) - 2, 0, -1):
            # iterate backwards until first layer
            backprop = np.dot(self.weights[i].T, error_cache[i + 1])
            error_cache[i] = backprop * activation.deriv(preactivation_cache[i])
            weight_grad_cache[i] = np.dot(error_cache[i], activation_cache[i - 1].T)

        # Gradient Descent
        for i in range(1, len(self.layer_sizes)):
            avg_weight_error_grad = weight_grad_cache[i] / len(inputs)
            avg_bias_error_grad = np.sum(error_cache[i], axis=1) / len(inputs)
            self.weights[i - 1] -= learning_rate * avg_weight_error_grad
            self.biases[i - 1] -= learning_rate * avg_bias_error_grad

    def batch_gradient_descent(self,
                               inputs,
                               outputs,
                               batch_size=10,
                               learning_rate=3,
                               cost_func=cross_entropy,
                               activation=sigmoid):
        rng_state = np.random.get_state()
        np.random.shuffle(inputs)
        np.random.set_state(rng_state)
        np.random.shuffle(outputs)

        input_batches = [
            inputs[k:k + batch_size]
            for k in range(0, len(inputs), batch_size)
        ]
        output_batches = [
            outputs[k:k + batch_size]
            for k in range(0, len(outputs), batch_size)
        ]

        for i in range(len(input_batches)):
            self.backpropagate(
                input_batches[i],
                output_batches[i],
                learning_rate,
                cost_func,
                activation,
            )

def main():
    network = Network([1, 16, 16, 1])
    inputs = np.linspace(-5, 5, 1000).reshape((1000, 1))
    outputs = np.sin(inputs) / inputs / 2 + 0.5

    import time
    start = time.perf_counter()
    for i in range(200):
        network.batch_gradient_descent(inputs, outputs)
    print("Finished training in", time.perf_counter() - start)

    network_output = network.propagate(inputs)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(inputs.reshape((1000,)), network_output.reshape((1000,)), c="r")
    ax1.scatter(inputs.reshape((1000,)), outputs.reshape((1000,)), c="b")
    plt.show()

if __name__ == "__main__":
    main()
