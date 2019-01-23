from read_data import read_data
from utils import *


# parametes
temperature = 10000
cooling_rate = 0.000003

cities, distances = read_data()
tour = generate_initial_solution(cities)
best = tour
while temperature > 1:
    print(temperature)
    new_tour = get_neighbour(tour)
    current_energy = get_distance(tour, distances)
    new_energy = get_distance(new_tour, distances)
    if random.uniform(0, 1) < acceptance_probability(current_energy, \
                                                    new_energy, temperature):
        tour = new_tour
    if get_distance(tour, distances) < get_distance(best, distances):
        best = tour
    temperature *= (1 - cooling_rate)

print(best)
print(get_distance(best, distances))
