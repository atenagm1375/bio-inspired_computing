import random
from numpy import exp
from copy import deepcopy


def acceptance_probability(current_energy, new_energy, temperature):
    if new_energy < current_energy:
        return 1.0
    return exp((current_energy - new_energy) / temperature)


def generate_initial_solution(cities):
    random.shuffle(cities)
    return cities


def get_distance(tour, distances):
    N = len(tour)
    dist = distances[tour[N - 1]][tour[0]]
    for i in range(1, N):
        dist += distances[tour[i - 1]][tour[i]]
    return dist


def get_neighbour(tour):
    N = len(tour)
    city1 = random.randint(0, N - 1)
    city2 = random.randint(0, N - 1)
    neighbor = deepcopy(tour)
    neighbor[city1], neighbor[city2] = neighbor[city2], neighbor[city1]
    return neighbor
