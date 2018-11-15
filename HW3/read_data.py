def read_data(folder="./DATA/", file_name="bayg29.tsp"):
    file = open(folder + file_name, 'r')
    for line in file:
        l = line.strip('\n').split("DIMENSION: ", 1)
        if len(l) > 1:
            n = int(l[1]) - 1
            break
    # n = int(file.readline().strip('\n').split("enter a dimention of matrix :", 1)[1])
    distances = [[0 for i in range(n+1)] for j in range(n+1)]
    cities = list(range(n))
    for line in file:
        if line.strip('\n') == "EDGE_WEIGHT_SECTION":
            print("yes")
            break
    for i in range(n):
        row = list(filter(('').__ne__, file.readline().strip("\n").split(' ')))[::-1]
        for j in range(len(row)):
            distances[i][j+1+i] = int(row[j])
            distances[j+1+i][i] = int(row[j])

    file.close()
    return cities, distances
