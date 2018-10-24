from read_data import read_data_2
from utils import initialize_population, compute_population_fitness, crossover, \
                    rank_based_selection, mutation, steady_state_replacement


# parameters
stagnancy_num = 20
p_c = 0.7
p_m = 0.05
p_rep = 0.25

# genetic algorithm process
cities, distances, num_routes = read_data_2()
population = initialize_population(cities, 100)
population_fitness = compute_population_fitness(population, distances)
cnt = 0
while cnt < stagnancy_num:
    mating_pool = rank_based_selection(population_fitness)
    children = crossover(mating_pool, p_c, "order")
    children = mutation(children, p_m, "swap")
    population = steady_state_replacement(population, children, p_rep)
    population_fitness = compute_population_fitness(population, distances)
