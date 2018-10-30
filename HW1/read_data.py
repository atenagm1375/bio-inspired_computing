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


def read_data_1(folder="./DATA/", file_name="Data1.txt"):
    file = open(folder + file_name, 'r')
    num_routes = 0
    distances = {}
    n = int(file.readline().strip('\n').split("enter a dimention of matrix :", 1)[1])
    cities = list(range(n))
    for line in file:
        if line.strip('\n') == "cost matix of city:":
            print("yes")
            break
    for i in range(n):
        row = list(filter(('').__ne__, file.readline().strip("\n").split(' ')))
        for j in range(len(row)):
            distances[(i, j)] = int(row[j])

    file.close()
    num_routes = len(distances)
    return cities, distances, num_routes
