import numpy as np

# each layer is an i*j array
# i = outputs, j = inputs

# Layer 1: 2 inputs, 3 outputs
weights1 = np.array([
    [-2, -2],
    [1, 0],
    [0, 1],
])
biases1 = np.array([
    [3],
    [0],
    [0]
])

# Layer 2: 3 inputs, 3 outputs
weights2 = np.array([
    [-2, -2, 0],
    [-2, 0, -2],
    [-4, 0, 0],
])
biases2 = np.array([
    [3],
    [3],
    [3]
])

# Layer 3: 3 inputs, 2 outputs
weights3 = np.array([
    [-2, -2, 0],
    [0, 0, 1],
])
biases3 = np.array([
    [3],
    [0]
])

weights = [weights1, weights2, weights3]
biases = [biases1, biases2, biases3]

def propagate(inputs):
    current_layer = np.array([inputs]).T
    for weight, bias in zip(weights, biases):
        weighted_input = np.dot(weight, current_layer) + bias
        current_layer = np.where(weighted_input <= 0, 0, 1)
    return current_layer.T[0]

sum, carry = propagate([1, 1])
print("Carry:", carry)
print("Sum:", sum)
