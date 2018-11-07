import numpy as np
from random import uniform, random


class Particle:
    def __init__(self, x0):
        self.x_i = x0
        self.v_i = [uniform(-1, 1) for i in range(dim)]
        self.pbest_i = []
        self.best_fitness_i = np.Inf
        self.fitness_i = np.Inf

    def evaluate(self, f):
        self.fitness_i = f(self.x_i)
        if self.fitness_i < self.best_fitness_i:
            self.pbest_i = self.x_i
            self.best_fitness_i = self.fitness_i

    def update_velocity(self, gbest, w, c1, c2):
        for i in range(dim):
            r1 = random()
            r2 = random()

            inertia = w * self.v_i[i]
            cognitive = c1 * r1 * (self.pbest_i[i] - self.x_i[i])
            social = c2 * r2 * (gbest[i] - self.x_i[i])
            self.v_i[i] = inertia + cognitive + social

    def update_position(self, bounds):
        for i in range(dim):
            self.x_i[i] = self.x_i[i] + self.v_i[i]
            if self.x_i[i] > bounds[i][1]:
                self.x_i[i] = bounds[i][1]
            elif self.x_i[i] < bounds[i][0]:
                self.x_i[i] = bounds[i][0]


class PSO:
    def __init__(self, f, bounds, w, c1, c2, N, max_iter):
        global dim

        dim = len(bounds)
        gbest_fitness = np.Inf
        gebest = []
        swarm = []

        # initialize swarm
        for i in range(N):
            swarm.append(Particle([uniform(bounds[j][0], bounds[j][1]) \
                                            for j in range(dim)]))

        for num_iter in range(max_iter):
            # evaluate fitness of particles and find gbest
            print("iteration number", num_iter + 1)
            for i in range(N):
                swarm[i].evaluate(f)
                if swarm[i].fitness_i < gbest_fitness:
                    gbest = swarm[i].x_i
                    gbest_fitness = swarm[i].fitness_i
            # update velocities and positions
            for i in range(N):
                swarm[i].update_velocity(gbest, w, c1, c2)
                swarm[i].update_position(bounds)
            print(gbest, gbest_fitness)

        print('-------------------------------------------------')
        print("result: f", tuple(gbest), '=', gbest_fitness)
