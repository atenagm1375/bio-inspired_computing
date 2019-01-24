import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from variables import *


def generate_dataset(n_classes=2):
    if n_classes == 2:
        X, y = make_moons(n_examples, noise=0.2)
        plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
        plt.show()
    return X, y
