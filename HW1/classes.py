import random
from copy import deepcopy


class City:
    def __init__(self, x, y, dist):
        self.x = x
        self.y = y
        self.distance = dist

    def __repr__(self):
        return "(" + str(self.x) + " ---" + str(self.distance) + "--> " + str(self.y) + ")"


def rank_based_selection(population, population_fitness):
    sum = len(population) * (len(population) + 1) / 2
    prob = [(i + 1) / sum for i in range(len(population))]
    return random.choices(population, weights=prob, k=len(population))


def steady_state_replacement(curr_generation, children, p_rep):
    next_generation = deepcopy(curr_generation)
    l = int(p_rep * len(next_generation))
    next_generation[:l] = children[-l:]
    return next_generation


class Chromosome(list):
    def __init__(self):
        list.__init__(self)
        self.distance = 0
        self.fitness = 0

    def __getitem__(self, key):
        return list.__getitem__(self, key)

    def __order_recombine_aid(self, point1, point2, child):
        i = point2 + 1
        while i != point1:
            if i >= len(self):
                i = 0
            j = i
            while True:
                if self[j] not in child:
                    child[i] = self[j]
                    break
                else:
                    j += 1
                    if j >= len(self):
                        j = 0
        return child

    def recombine(self, type="order", ind_2):
        child1 = [-1] * len(self)
        child2 = [-1] * len(self)

        if type == "order":
            point1 = random.randint(0, len(self))
            point2 = random.randint(0, len(self))
            if point2 < point1:
                point1, point2 = point2, point1

            child1[point1:point2] = self[point1:point2]
            child2[point1:point2] = ind_2[point1:point2]

            child1 = ind_2.__order_recombine_aid(point1, point2, child1)
            child2 = self.__order_recombine_aid(point1, point2, child2)
        elif type == "cycle":
            cycles = []
            for j in range(len(self)):
                i = j
                if j > 0 and i in cycles[j - 1]:
                    continue
                cycle = []
                while True:
                    if i in cycle:
                        break
                    cycle.append(i)
                    i = self.index(ind_2[i])
                if cycle:
                    cycles.append(cycle)

            flag = True
            for cycle in cycles:
                if flag:
                    for i in cycle:
                        child1[i] = self[i]
                    flag = False
                else:
                    for i in cycle:
                        child2[i] = ind_2[i]
                    flag = True

        return child1, child2

        def mutate(self, type="swap"):
            child = deepcopy(self)
            s1 = random.randint(0, len(child))
            s2 = random.randint(0, len(child))
            if type == "swap":
                child[s1], child[s2] = child[s2], child[s1]
            elif type == "insert":
                if s1 > s2:
                    s2, s1 = s1, s2
                tmp = child[s2]
                for i in range(s1 + 2, s2):
                    child[i] = child[i - 1]
                child[s1 + 1] = tmp
            elif type == "scramble":
                if s1 > s2:
                    s2, s1 = s1, s2
                child[s1:s2] = random.shuffle(child[s1:s2])
            elif type == "inversion":
                if s1 > s2:
                    s2, s1 = s1, s2
                mid = (s2 - s1) // 2
                for i in range(s1, s1 + mid):
                    child[s1], child[s1 + mid - i] = child[s1 + mid - i] + child[i]

            return child

        def fitness(self):
            if self.distance == 0:
                dist = 0
                for i in range(0, len(self)):
                    fromCity = self[i]
                    toCity = None
                    if i + 1 < len(self):
                        toCity = self[i + 1]
                    else:
                        toCity = self[0]
                    dist += fromCity.distance(toCity)
                self.distance = dist
                if self.fitness == 0:
                    self.fitness = 1 / float(self.distance)
            return self.fitness
