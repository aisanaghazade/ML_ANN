import fnmatch
import os
import HOG
import ChiSquare
import numpy as np
import sys
import K_means as K
import math
import time
import pandas as pd

img_infos = {} #the owner of each image
ref_imgs = []
img = HOG.HOG('test/asamma.20.jpg') #input image
NC = 2#number of clusters
treshhold = 0.09
x = 0 #this variable is used in probability of each class
weights = np.zeros(NC) #weights for each cluster
CDistances = np.zeros(NC) #dictances between clusters and input image
classes = {} #classes in reference images
print(1)
distances = {}
for i in range(0, NC):
    distances[str(i)] = []

#reading files
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

clusters, centroids = K.KMeans_clustering(ref_imgs, NC,  img_infos) #clustering reference images by K-Means algorithm
print(2)

#calculating distances between reference images
distance_matrix = []
# for i in range (0, len(ref_imgs)):
#     t1 = time.time()
#     distance = []
#     for j in range (0, len(ref_imgs)):
#         if i == j:
#             distance.append(0)
#         else:
#             distance.append(ChiSquare.distance_computation( ref_imgs[i], ref_imgs[j]))
#     distance_matrix.append(distance)
#     print(img_infos[i])
#     print(time.time() - t1)
#
# np.savetxt("distances.csv", distance_matrix, delimiter=",")

# distance_matrix = pd.read_csv("distances.csv")
# distance_matrix = np.array(distance_matrix)
distance_matrix = np.genfromtxt('distances.csv', delimiter = ',')
print(3)
queues = {}
queue = []
for i in range(0, NC):
    queues[str(i)] = []

itr = 0 #iterations

for i in range(0, NC):
    queues[str(i)].append(centroids[i])
    queue.append(centroids[i])
    clusters[str(i)].remove(centroids[i])

nearest = queues[str(0)][0]
min_dist = ChiSquare.distance_computation(ref_imgs[queues[str(0)][0]], img)
distances[str(0)].append(min_dist)
max_dist = min_dist
CDistances[0] += min_dist


for i in range(1, NC):
    distance = ChiSquare.distance_computation(ref_imgs[queues[str(i)][0]], img)
    distances[str(i)].append(distance)
    CDistances[i] += distance
    if distance < min_dist:
        nearest = queues[str(i)][0]
        min_dist = distance

    if distance > max_dist:
        max_dist = distance

weights = CDistances/(max_dist)
weights = np.power(weights, -1)
for i in range(0, len(weights)):
    weights[i] = math.ceil(weights[i])




x = 0

print(4)

t = time.time()
isEmpty = False


while True:
    print(weights)
    print(5)


    # for j in range(0, len(queues)):
    #     for i in range(0, len(queues[str(j)])):
    #         x += 1
    #         distance = ChiSquare.distance_computation(ref_imgs[queues[str(j)][i]], img)
    #         distances[str(j)].append(distance)
    #         if distance <= treshhold:
    #             print("distance:" + str(distance))
    #             print(img_infos[queues[str(j)][i]])
    #             print(time.time() - t)
    #             sys.exit()
    #
    #         if distance < min_dist:
    #             min_dist = distance
    #             nearest = queues[str(j)][i]
    print(6)

    if min_dist <= treshhold:
        print("distance:" + str(min_dist))
        print(img_infos[nearest])
        print(time.time() - t)
        sys.exit()

    if itr > 100 or isEmpty:
        print(queues)
        print("The nearest class is: "+ img_infos[nearest])
        print(time.time() - t)
        sys.exit()

    itr += 1
    argument = 0
    min = sys.maxsize
    print(7)
    isEmpty = True
    for j in range(0, len(clusters)):
        # if(weights[j] <= 1):
        #     continue
        for w in range(0, math.ceil(weights[j])*math.ceil(weights[j])):
            argument = 0
            min = sys.maxsize
            for i in range(0, len(clusters[str(j)])):
                # print(8)
                likelihood = 0

                for r_j in range(0, len(clusters)):
                    for r_i in range(0, len(queues[str(r_j)])):
                        print(distances)
                        print(r_i)
                        print(r_j)
                        print(queues)
                        print(len(distance_matrix))
                        print(len(distance_matrix[0]))
                        fi = ((distances[str(r_j)][r_i] - distance_matrix[queues[str(r_j)][r_i]][clusters[str(r_j)][i]]) ** 2) / \
                             distance_matrix[queues[str(r_j)][r_i]][clusters[str(r_j)][i]]
                        likelihood += fi
                p = classes[img_infos[clusters[str(j)][i]]]
                # print(p)
                likelihood = likelihood - math.log(p)
                if likelihood <= min:
                    min = likelihood
                    argument = i

            if len(clusters[str(j)]) > 0:
                distance = ChiSquare.distance_computation(ref_imgs[clusters[str(j)][argument]], img)
                if distance < min_dist:
                    min_dist = distance
                    nearest = clusters[str(j)][argument]
                CDistances[j] += distance
                distances[str(j)].append(distance)
                queues[str(j)].append(clusters[str(j)][argument])
                queue.append(clusters[str(j)][argument])
                del (clusters[str(j)][argument])
                isEmpty = False

    # max_dist = 0
    # avg_distances = []
    # for i in range(0, len(CDistances)):
    #     avg_distances.append(CDistances[i]/len(queues[str(i)]))
    #     if avg_distances[i] >= max_dist:
    #         max_dist = avg_distances[i]

    # weights = avg_distances / (max_dist)
    # weights = np.power(weights, -1)
    # for i in range(0, len(weights)):
    #     weights[i] = math.ceil(weights[i])

