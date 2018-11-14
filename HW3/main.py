from read_data import read_data
from utils import initialize_population, compute_population_fitness, crossover, \
                    rank_based_selection, mutation, steady_state_replacement


# parameters
stagnancy_num = 25
p_c = 0.8
p_m = 0.1
p_rep = 0.25
population_num = 1000
iteration_num = 1
cross_over_type = "order"
mutation_type = "inversion"


# genetic algorithm process
cities, distances, num_routes = read_data()
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
    children = mutation(children, p_m, mutation_type)
    children_fitness = compute_population_fitness(children, distances)
    population = steady_state_replacement(population, population_fitness, \
                                            children,children_fitness, p_rep)
    # print(population)

    iteration_num += 1

print("******************************************")
print("best distance found:", 1 / fittest[1])
print("best route found:", population[fittest[0]])
