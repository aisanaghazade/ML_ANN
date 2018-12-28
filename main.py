import fnmatch
import os
import HOG
import ChiSquare
import numpy as np
import pandas as pd
import cv2
import sys
import K_means as K
import random
import math
import time

img_infos = {}
ref_imgs = []
img = HOG.HOG('test/asamma.20.jpg')

treshhold = 0.1
x = 0
classes = {}
print(1)
for dirpath, dirs, files in os.walk('faces94'):
    for filename in fnmatch.filter(files, '*.jpg'):
        with open(os.path.join(dirpath, filename)):
            img_infos[x] = str(filename.split('.')[0])
            x += 1
            # ref_imgs.append(cv2.imread(os.path.join(dirpath, filename)))
            ref_imgs.append(HOG.HOG(os.path.join(dirpath, filename)))
            if str(filename.split('.')[0]) in classes:
                classes[str(filename.split('.')[0])] += 1.0
            else:
                classes[str(filename.split('.')[0])] = 1.0
    print(dirpath)
for Class in classes:
    if classes[Class] == 0:
        print(Class)
    classes[Class] /= len(ref_imgs)

print(2)
distance_matrix = []
for i in range (0, len(ref_imgs)):
    t1 = time.time()
    distance = []
    for j in range (0, len(ref_imgs)):
        if i == j:
            distance.append(0)
        else:
            distance.append(ChiSquare.distance_computation( ref_imgs[i], ref_imgs[j]))
    distance_matrix.append(distance)
    print(img_infos[i])
    print(time.time() - t1)

np.savetxt("distances.csv", distance_matrix, delimiter=",")

# distance_matrix = pd.read_csv("distances.csv")
print(3)
queue = []
itr = 0

queue.append(0)
min_dist = ChiSquare.distance_computation(ref_imgs[queue[0]], img)
nearest = queue[0]
R = list(range(0, len(distance_matrix)))
x = 0
del(R[nearest])
print(4)

t = time.time()
while True:
    print(5)
    distances = []
    for i in range(0, len(queue)):
        x += 1
        distance = ChiSquare.distance_computation(ref_imgs[queue[i]], img)
        distances.append(distance)
        if distance <= treshhold:
            print("distance:"+ str(distance))
            print(img_infos[queue[i]])
            print(time.time() - t)
            sys.exit()


        if distance < min_dist:
            min_dist = distance
            nearest = queue[i]
    print(6)
    if itr > 100:
        print(queue)
        print("The nearest class is: "+ img_infos[nearest])
        print(time.time() - t)
        sys.exit()

    itr += 1
    argument = 0
    min = sys.maxsize
    print(7)
    for i in range(0, len(R)):
        # print(8)
        likelihood = 0
        for r_i in range(0, len(queue)):
            # print(queue[r_i])
            # print(R[i])
            fi = ((distances[r_i] - distance_matrix[queue[r_i]][R[i]]) ** 2) / distance_matrix[queue[r_i]][R[i]]
            likelihood += fi
        p = classes[img_infos[R[i]]]
        # print(p)
        likelihood = likelihood - math.log(p)
        if likelihood <= min:
            min = likelihood
            argument = i

    queue.append(R[argument])
    del(R[argument])