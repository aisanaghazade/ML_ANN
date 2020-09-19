import numpy as np
import math
distance = 0
# we use Chi-square distance computation.
def distance_computation(img1, img2):
    distance = 0.0
    distances = []
    for i in range (0, len(img1)):
        dist = []
        for j in range(0, len(img1[0])):
            d = 0
            for k in range(0, len(img1[0][0])):
                if(img1[i][j][k] != 0 or img2[i][j][k] != 0):
                    d += ((img1[i][j][k]-img2[i][j][k])**2)/(img1[i][j][k]+img2[i][j][k])
            d /= 2
            if math.isnan(distance):
                print(d)
                return 0

            distance =distance + d
            dist.append(d)
        distances.append(dist)
    distance /= len(img1[0]) * len(img1)
    return distance

