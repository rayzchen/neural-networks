import numpy as np

# each layer is a (i+1)*(j+1) array
# i = outputs, j = inputs
# last column is the biases
# last row is all zeros except the last value which is 1
# the input column vector has 1 as its last value

# Layer 1: 2 inputs, 3 outputs
layer1 = np.array([
    [-2, -2, 3],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Layer 2: 3 inputs, 3 outputs
layer2 = np.array([
    [-2, -2, 0, 3],
    [-2, 0, -2, 3],
    [-4, 0, 0, 3],
    [0, 0, 0, 1]
])

# Layer 3: 3 inputs, 2 outputs
layer3 = np.array([
    [-2, -2, 0, 3],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

layers = [layer1, layer2, layer3]

def propagate(inputs):
    current_layer = np.array([[*inputs, 1]]).T
    for layer in layers:
        values = np.dot(layer, current_layer)
        current_layer = np.where(values <= 0, 0, 1)
    return current_layer.T[0,:-1]

sum, carry = propagate([1, 1])
print("Carry:", carry)
print("Sum:", sum)
