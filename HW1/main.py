from read_data import read_data_2
from utils import initialize_population, compute_population_fitness, crossover, \
                    rank_based_selection, mutation, steady_state_replacement


# parameters
stagnancy_num = 20
p_c = 0.7
p_m = 0.05
p_rep = 0.25
population_num = 100
iteration_num = 1

# genetic algorithm process
cities, distances, num_routes = read_data_2()
population = initialize_population(cities, population_num)
# population_fitness = compute_population_fitness(population, distances)
# fittest = population_fitness[-1]
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
    children = crossover(population, mating_pool, p_c, "order")
    children = mutation(children, p_m, "insert")
    children_fitness = compute_population_fitness(children, distances)
    population = steady_state_replacement(population, population_fitness, \
                                            children,children_fitness, p_rep)
    # print(population)

    iteration_num += 1

print("******************************************")
print("best distance found:", 1 / fittest[1])
print("best route found:", population[fittest[0]])
