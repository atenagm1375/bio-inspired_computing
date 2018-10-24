from classes import City

def read_data_2(folder="./DATA/", file_name="Data2.txt"):
    file = open(folder + file_name, 'r')
    num_routes = file.readline()
    cities = []
    for line in file:
        if line != '\n':
            d = line.split()
            cities.append(City(d[0], d[1], d[2]))
    file.close()
    return cities, num_routes


c, n = read_data_2()
print(c)
