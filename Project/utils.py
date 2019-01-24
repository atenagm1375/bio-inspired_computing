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


def calculate_loss(model, X):
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
