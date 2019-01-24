import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from variables import *


def generate_dataset(n_classes=2):
    if n_classes == 2:
        np.random.seed(0)
        X, y = make_moons(n_examples, noise=0.2)
        plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
        plt.show()
    return X, y


def activation(func, z):
    if func == "tanh":
        return np.tanh(z)
    elif func == "softmax":
        exp_score = np.exp(z)
        return exp_score / np.sum(exp_score, axis=1, keepdims=True)


def forward_propagation(X, W, b, func):
    z = X.dot(W) + b
    return activation(func, z)


def calculate_loss(model, X, y):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]

    # forward propagation to hidden layer
    a1 = forward_propagation(X, W1, b1, activation_function1)
    # foward propagation to output layer
    a2 = forward_propagation(a1, W2, b2, activation_function2)

    logprobs = -np.log(a2[range(n_examples), y])
    data_loss = np.sum(logprobs)
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / n_examples * data_loss


def predict(model, X):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]

        # forward propagation to hidden layer
        a1 = forward_propagation(X, W1, b1, activation_function1)
        # foward propagation to output layer
        a2 = forward_propagation(a1, W2, b2, activation_function2)

        return np.argmax(a2, axis=1)


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False, batch_gd=False):
    np.random.seed(0)
    W1 = np.random.randn(n_input_dim, nn_hdim) / np.sqrt(n_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, n_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, n_output_dim))

    model = {}

    for i in xrange(0, num_passes):
        # forward propagation
        a1 = activation(X, W1, b1, activation_function1)
        a2 = activation(a1, W2, b2, activation_function2)

        # backpropagation
        # TODO: put it in a separate function
        delta3 = a2
        delta3[range(n_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # add regularization terms
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # update parameters
        W1 += -lr * dW1
        b1 += -lr * db1
        W2 += -lr * dW2
        b2 += -lr * db2

        model = {"W1": W1, "b1": b1, "W2": W2, "b2":b2}

        if print_loss and i % 1000 == 0:
            print("Loss after iteration {}: {}".format(i, calculate_loss(model, X, y)))

    return model
