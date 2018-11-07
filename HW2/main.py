from functions import func1, func2, bounds
from pso import PSO

# initializtion
w = 0.5
c1 = 1
c2 = 2
N = 50
max_iter = 1000
stagnancy = 100

PSO(func1, bounds[0], w, c1, c2, N, max_iter, stagnancy)
