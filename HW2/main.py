from functions import func1, func2, bounds
from pso import PSO

# initializtion
w = 0.2
c1 = 1.5
c2 = 2.5
N = 50
max_iter = 1000
stagnancy = 50

PSO(func2, bounds[1], w, c1, c2, N, max_iter, stagnancy)
