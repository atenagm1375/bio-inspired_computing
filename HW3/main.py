from read_data import read_data
from utils import initialize_population, compute_population_fitness, crossover, \
                    rank_based_selection, mutation, steady_state_replacement, \
                    local_search


# parameters
stagnancy_num = 25
p_c = 0.6
p_m = 0.2
p_rep = 0.25
p_local = 0.5
population_num = 1000
neighbor_num = 10
iteration_num = 1
cross_over_type = "order"
mutation_type = "inversion"
isLamarck = False


# memetic algorithm process
cities, distances = read_data()
population = initialize_population(cities, population_num)
fittest = (len(population), 0)
cnt = 0
while True:
    population_fitness = compute_population_fitness(population, distances)
    if cnt >= stagnancy_num:
        break
    elif fittest[1] == population_fitness[-1][1]:
        cnt += 1
    else:
        cnt = 0
    fittest = population_fitness[-1]
    print("iteration number", iteration_num, ":")
    print("best distance of generation:", 1 / fittest[1])
    mating_pool = rank_based_selection(dict(population_fitness))
    children = crossover(population, mating_pool, p_c, cross_over_type)
    children_fitness = compute_population_fitness(children, distances)
    local_search(children, children_fitness, distances, p_local, neighbor_num, isLamarck)
    children = mutation(children, p_m, mutation_type)
    children_fitness = compute_population_fitness(children, distances)
    local_search(children, children_fitness, distances, p_local, neighbor_num, isLamarck)
    children_fitness = sorted(children_fitness, key=lambda x: x[1])
    population = steady_state_replacement(population, population_fitness, \
                                            children,children_fitness, p_rep)
    # print(population)

    iteration_num += 1

print("******************************************")
print("best distance found:", 1 / fittest[1])
print("best route found:", population[fittest[0]])
