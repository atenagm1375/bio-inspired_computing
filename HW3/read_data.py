def read_data(folder="./DATA/", file_name="bayg29.tsp"):
    file = open(folder + file_name, 'r')
    num_routes = 0
    distances = {}
    for line in file:
        l = line.strip('\n').split("DIMENSION: ", 1)
        if len(l) > 1:
            n = int(l[1]) - 1
            break
    # n = int(file.readline().strip('\n').split("enter a dimention of matrix :", 1)[1])
    cities = list(range(n))
    for line in file:
        if line.strip('\n') == "EDGE_WEIGHT_SECTION":
            print("yes")
            break
    for i in range(n):
        row = list(filter(('').__ne__, file.readline().strip("\n").split(' ')))
        for j in range(len(row)):
            distances[(i, j)] = int(row[j])

    file.close()
    num_routes = len(distances)
    return cities, distances, num_routes
