from functions import *
from random import uniform, random


class Particle:
    def __init__(self, x0):
        self.x_i = x0
        self.dim = len(self.x_i)
        self.v_i = [uniform(-1, 1) for i in range(self.dim)]
        self.pbest_i = []
        self.best_fitness_i = np.Inf
        self.fitness_i = np.Inf

    def evaluate(self, f):
        self.fitness_i = f(self.x_i[0], self.x_i[1])
        if self.fitness_i < self.best_fitness_i:
            self.pbest_i = x_i
            self.best_fitness_i = self.fitness_i

    def update_velocity(self, gbest, w, c1, c2):
        for i in range(self.dim):
            r1 = random()
            r2 = random()

            inertia = w * self.v_i[i]
            cognitive = c1 * r1 * (self.pbest_i - self.x_i)
            social = c2 * r2 * (gbest - self.x_i)
            self.v_i[i] = inertia + cognitive + social

    def update_position(self, bounds):
        for i in range(self.dim):
            self.x_i[i] = self.x_i[i] + self.v_i[i]
            if self.x_i[i] > bounds[i][1]:
                self.x_i = bounds[i][1]
            elif self.x_i[i] < bounds[i][0]:
                self.x_i = bounds[i][0]
