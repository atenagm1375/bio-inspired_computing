from read_data import read_data_2
from utils import initialize_population, compute_population_fitness, crossover, \
                    rank_based_selection, mutation, steady_state_replacement


# parameters
stagnancy_num = 20
p_c = 0.7
p_m = 0.05
p_rep = 0.25
population_num = 10
iteration_num = 1

# genetic algorithm process
cities, distances, num_routes = read_data_2()
population = initialize_population(cities, population_num)
# population_fitness = compute_population_fitness(population, distances)
# fittest = population_fitness[-1]
cnt = 0
while True:
    print("iteration number", iteration_num, ":")
    population_fitness = compute_population_fitness(population, distances)
    fittest = population_fitness[-1]
    mating_pool = rank_based_selection(dict(population_fitness))
    children = crossover(population, mating_pool, p_c, "order")
    children = mutation(children, p_m, "swap")
    children_fitness = compute_population_fitness(children, distances)
    population = steady_state_replacement(population, population_fitness, \
                                            children,children_fitness, p_rep)
    # print(population)
    if cnt >= stagnancy_num:
        break
    elif fittest == population_fitness[-1]:
        cnt += 1
    else:
        cnt = 0

    fittest = population_fitness[-1]
    print("best fitness of new generation:", fittest[1])
    iteration_num += 1

print("******************************************")
print("best fitness found:", fittest[1])
print("best route found:", population[fittest[0]])
