def read_data_2(folder="./DATA/", file_name="Data2.txt"):
    file = open(folder + file_name, 'r')
    num_routes = file.readline()
    distances = {}
    cities = []
    for line in file:
        if line != '\n':
            d = line.split()
            if int(d[2]) != 0:
                distances[(int(d[0]), int(d[1]))] = int(d[2])
                if int(d[0]) not in cities:
                    cities.append(int(d[0]))

    file.close()
    return cities, distances, num_routes
