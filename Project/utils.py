import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification


def generate_dataset(n_classes=2, n_examples=200, n_features=2):
    if n_classes == 2:
        np.random.seed(0)
        X, y = make_moons(n_examples, noise=0.2)
    else:
        np.random.seed(0)
        X, y = make_classification(n_samples=n_examples, n_features=n_features, \
                                    n_repeated=0, n_redundant=0, n_clusters_per_class=1, n_classes=n_classes)
    return X, y


def plot_dataset(X, y, n_classes):
    plt.scatter(X[:, 0], X[:, 1], s=20, c=y, cmap=plt.cm.Spectral)
    plt.show()


class NeuralNetwork:
    def __init__(self, func, X, y, n_examples, n_input_dim, n_classes):
        self.activation_functions = func
        self.X = X
        self.y = y
        self.n_examples = n_examples
        self.n_input_dim = n_input_dim
        self.n_output_dim = n_classes


    def activation(self, func, z):
        if func == "softmax":
            exp_score = np.exp(z)
            return exp_score / np.sum(exp_score, axis=1, keepdims=True)
        elif func == "tanh":
            return np.tanh(z)
        elif func == "sigmoid":
            return 1. / (1 + np.exp(-z))
        elif func == "ReLU":
            return np.maximum(z, np.zeros(z.shape))
        elif func == "leaky ReLU":
            return np.maximum(z, 0.1 * z)


    def activation_grad(self, func, z):
        if func == "softmax":
            pass
        elif func == "tanh":
            return 1 - np.square(np.tanh(z))
        elif func == "sigmoid":
            s = 1. / (1 + np.exp(-z))
            return s * (1 - s)
        elif func == "ReLU":
            z[z <= 0] = 0
            z[z > 0] = 1
            return z
        elif func == "leaky ReLU":
            z[z <= 0] = 0.01
            z[z > 0] = 1
            return z


    def forward_propagation(self, X, W, b, func):
        z = X.dot(W) + b
        return z, self.activation(func, z)


    def calculate_loss(self, model, reg_lambda):
        W1, b1, W2, b2, W3, b3 = model["W1"], model["b1"], model["W2"], model["b2"], model["W3"], model["b3"]

        # forward propagation to hidden layer
        z1, a1 = self.forward_propagation(self.X, W1, b1, self.activation_functions[0])
        # foward propagation to second hidden layer
        z2, a2 = self.forward_propagation(a1, W2, b2, self.activation_functions[0])
        # forward propagation to ouptput layer
        z3, a3 = self.forward_propagation(a2, W3, b3, self.activation_functions[1])

        logprobs = -np.log(a3[range(self.n_examples), self.y])
        data_loss = np.sum(logprobs)
        data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
        return 1. / self.n_examples * data_loss


    def predict(self, model, x):
        W1, b1, W2, b2, W3, b3 = model["W1"], model["b1"], model["W2"], model["b2"], model["W3"], model["b3"]

        # forward propagation to hidden layer
        z1, a1 = self.forward_propagation(x, W1, b1, self.activation_functions[0])
        # foward propagation to second hidden layer
        z2, a2 = self.forward_propagation(a1, W2, b2, self.activation_functions[0])
        # forward propagation to output layer
        z3, a3 = self.forward_propagation(a2, W3, b3, self.activation_functions[1])

        return np.argmax(a3, axis=1)


    def build_model(self, nn_hdim, num_passes=20000, lr=0.01, print_loss=False, \
                    minibatch_size=200, reduce_lr=False, decay=0.01, \
                    reg_lambda=0.01):
        np.random.seed(0)
        W1 = np.random.randn(self.n_input_dim, nn_hdim[0]) / np.sqrt(self.n_input_dim)
        b1 = np.zeros((1, nn_hdim[0]))
        W2 = np.random.randn(nn_hdim[0], nn_hdim[1]) / np.sqrt(nn_hdim[0])
        b2 = np.zeros((1, nn_hdim[1]))
        W3 = np.random.randn(nn_hdim[1], self.n_output_dim) / np.sqrt(nn_hdim[1])
        b3 = np.zeros((1, self.n_output_dim))

        model = {}

        for i in range(num_passes):
            for j in range(0, self.X.shape[0], minibatch_size):
                X_train = self.X[j:j + minibatch_size]
                y_train = self.y[j:j + minibatch_size]
                # forward propagation
                z1, a1 = self.forward_propagation(X_train, W1, b1, self.activation_functions[0])
                z2, a2 = self.forward_propagation(a1, W2, b2, self.activation_functions[0])
                z3, a3 = self.forward_propagation(a2, W3, b3, self.activation_functions[1])

                # backpropagation
                delta3 = a3
                delta3[range(minibatch_size), y_train] -= 1
                dW3 = (a2.T).dot(delta3)
                db3 = np.sum(delta3, axis=0, keepdims=True)
                delta2 = delta3.dot(W3.T) * self.activation_grad(self.activation_functions[0], z2)
                dW2 = (a1.T).dot(delta2)
                db2 = np.sum(delta2, axis=0)
                delta1 = delta2.dot(W2.T) * self.activation_grad(self.activation_functions[0], z1)
                dW1 = X_train.T.dot(delta1)
                db1 = np.sum(delta1, axis=0)

                # add regularization terms
                dW3 += reg_lambda * W3
                dW2 += reg_lambda * W2
                dW1 += reg_lambda * W1

                # update parameters
                W1 += -lr * dW1
                b1 += -lr * db1
                W2 += -lr * dW2
                b2 += -lr * db2
                W3 += -lr * dW3
                b3 += -lr * db3

                # update learning rate
                if reduce_lr:
                    lr *= 1 / (1 + decay * i)

                model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

            if print_loss and i % 1000 == 0:
                print("Loss after iteration {}: {}".format(i, self.calculate_loss(model, reg_lambda)))

        return model


    def plot_decision_boundary(self, model, plot_name):
        # Set min and max values and give it some padding
        x_min, x_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
        y_min, y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the function value for the whole grid
        Z = self.predict(model, np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.Spectral)
        # plt.show()
        plt.savefig(plot_name)
