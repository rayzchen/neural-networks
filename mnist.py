import pickle
import gzip
import numpy as np
from backprop_activations import Network

def load_data():
    f = gzip.open("mnist.pkl.gz", "rb")
    tr_d, va_d, te_d = pickle.load(f, encoding="latin1")
    f.close()
    training_inputs = tr_d[0]
    training_results = np.array([vectorized_result(y) for y in tr_d[1]])
    training_data = (training_inputs, training_results)
    validation_inputs = va_d[0]
    validation_results = np.array([vectorized_result(y) for y in va_d[1]])
    validation_data = (validation_inputs, validation_results)
    test_inputs = te_d[0]
    test_results = np.array([vectorized_result(y) for y in te_d[1]])
    test_data = (test_inputs, test_results)
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10,))
    e[j] = 1.0
    return e

def main():
    training_data, validation_data, test_data = load_data()
    network = Network([784, 30, 10])

    test_accuracy = []
    training_accuracy = []
    for i in range(60):
        train_inputs, train_outputs = training_data
        network.batch_gradient_descent(train_inputs, train_outputs, regularization=2.0)

        actual_outputs = np.argmax(network.propagate(training_data[0]), axis=1)
        expected_outputs = np.argmax(training_data[1], axis=1)
        evaluation = np.sum(actual_outputs == expected_outputs)
        training_accuracy.append(evaluation / len(training_data[0]) * 100)

        actual_outputs = np.argmax(network.propagate(test_data[0]), axis=1)
        expected_outputs = np.argmax(test_data[1], axis=1)
        evaluation = np.sum(actual_outputs == expected_outputs)
        test_accuracy.append(evaluation / len(test_data[0]) * 100)

        print("Epoch", i + 1, "| Test accuracy:", round(test_accuracy[-1], 2))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(60), test_accuracy)
    ax1.plot(np.arange(60), training_accuracy)
    plt.show()

if __name__ == "__main__":
    main()
