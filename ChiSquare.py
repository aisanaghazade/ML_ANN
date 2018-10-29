import numpy as np
distance = 0
def distance_computation(img1, img2):
    distance = 0.0
    # print(distance)
    distances = []
    for i in range (0, len(img1)):
        dist = []
        for j in range(0, len(img1[0])):
            d = 0
            for k in range(0, len(img1[0][0])):
                d += ((img1[i][j][k]-img2[i][j][k])**2)/(img1[i][j][k]+img2[i][j][k])
            d /= 2
            distance =distance + d
            dist.append(d)
        distances.append(dist)
    distance = 0.0
    for i in range(0, len(distances)):
        for j in range(0, len(distances[0])):
            distance += distances[i][j]
    distance /= len(img1[0]) * len(img1)
    # distance = np.mean(distances)
    print(distance)
    # print(distances)

