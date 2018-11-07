import numpy as np


def func1(x, y):
    z = np.abs(1 - np.sqrt(x ** 2 + y ** 2) / np.pi)
    return - np.abs(np.sin(x) * np.cos(y) * np.exp(z))

def func2(x, y):
    z = np.abs(x ** 2 - y ** 2)
    z = np.cos(np.sin(z)) ** 2 - 0.5
    return 0.5 + z / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2

bounds = [((-10, 10), (-10, 10)), {((-100, 100), (-100, 100))}]
