import random
from copy import deepcopy
import operator
from itertools import chain


def initialize_population(cities, N):
    population = []
    for i in range(N):
        population.append(random.sample(cities, len(cities)))
    return population


def compute_population_fitness(population, distances):
    fitnesses = {}
    for i in range(len(population)):
        fitnesses[i] = compute_fitness(population[i], distances)
    return sorted(fitnesses.items(), key=operator.itemgetter(1))


def compute_fitness(route, distances):
    dist = 0

    for i in range(1, len(route)):
        if (route[i - 1], route[i]) in distances.keys():
            dist += distances[(route[i - 1], route[i])]
        elif (route[i], route[i - 1]) in distances.keys():
            dist += distances[(route[i], route[i - 1])]
        else:
            dist += (2 * max(list(distances.values())))
    return 1 / float(dist) if dist != 0 else 0


def rank_based_selection(population):
    sum = len(population) * (len(population) + 1) / 2
    prob = [(i + 1) / sum for i in range(len(population))]
    return random.choices(list(population.keys()), weights=prob, k=len(population))


def steady_state_replacement(curr_generation, pop_fit, children, chld_fit, p_rep):
    next_generation = [curr_generation[pop[0]] for pop in pop_fit]
    l = int(p_rep * len(next_generation))
    chld = chld_fit[-l:]
    next_generation[:l] = [children[pop[0]] for pop in chld]
    return next_generation
    # next_generation = deepcopy(curr_generation)
    # l = int(p_rep * len(next_generation))
    # next_generation[:l] = children[-l:]
    # return next_generation


def crossover(population, pool, p_c, type="order"):
    children = []
    n = len(pool)
    for i in range(n // 2):
        child1 = population[pool[i]]
        child2 = population[pool[n - i - 1]]
        if random.uniform(0, 1) < p_c:
            child1, child2 = recombine(child1, child2, type)
        children.append(child1)
        children.append(child2)
    return children


def order_recombine_aid(par, point1, point2, child):
    i = point2
    while i != point1:
        if i >= len(par):
            if point1 != 0:
                i = 0
            else:
                break
        j = i
        while True:
            if par[j] not in child:
                child[i] = par[j]
                break
            else:
                j += 1
                if j >= len(par):
                    j = 0
        i += 1
    return child

def recombine(par1, par2, type):
    child1 = [-1] * len(par1)
    child2 = [-1] * len(par2)

    if type == "order":
        point1 = random.randint(2, len(par1)) - 2
        point2 = random.randint(1, len(par1)) - 1
        if point2 < point1:
            point1, point2 = point2, point1

        child1[point1:point2] = par1[point1:point2]
        child2[point1:point2] = par2[point1:point2]

        # print(point1, point2)
        # print(par1, par2)
        child1 = order_recombine_aid(par2, point1, point2, child1)
        child2 = order_recombine_aid(par1, point1, point2, child2)
    elif type == "cycle":
        cycles = []
        for j in range(len(par1)):
            i = j
            if i in chain.from_iterable(cycles):
                continue
            cycle = []
            while True:
                if i in cycle:
                    break
                cycle.append(i)
                i = par1.index(par2[i])
            if cycle:
                cycles.append(cycle)

        flag = True
        for cycle in cycles:
            if flag:
                for i in cycle:
                    child1[i] = par1[i]
                    child2[i] = par2[i]
                flag = False
            else:
                for i in cycle:
                    child2[i] = par1[i]
                    child1[i] = par2[i]
                flag = True

    return child1, child2


def mutation(children, p_m, type="swap"):
    for i in range(len(children)):
        if random.uniform(0, 1) < p_m:
            children[i] = mutate(children[i], type)
    return children


def mutate(child, type):
    child = deepcopy(child)
    s1 = random.randint(2, len(child)) - 2
    s2 = random.randint(1, len(child)) - 1
    while s1 == s2:
        s2 = random.randint(1, len(child)) - 1
    if type == "swap":
        child[s1], child[s2] = child[s2], child[s1]
    elif type == "insert":
        if s1 > s2:
            s2, s1 = s1, s2
        # print(s1, s2)
        tmp = child[s2]
        for i in range(s2, s1 + 1, -1):
            child[i] = child[i - 1]
        child[s1 + 1] = tmp
        # print("child:", child)
    elif type == "scramble":
        if s1 > s2:
            s2, s1 = s1, s2
        copy = child[s1:s2]
        random.shuffle(copy)
        child[s1:s2] = copy
    elif type == "inversion":
        if s1 > s2:
            s2, s1 = s1, s2
        mid = (s2 - s1) // 2 + s1
        # print(s1, s2, mid)
        # print(child)
        for i in range(s1, mid):
            child[i], child[s2 - i + s1] = child[s2 - i + s1], child[i]
        # print(child)

    return child
