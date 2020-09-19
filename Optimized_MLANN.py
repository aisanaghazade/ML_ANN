import fnmatch
import os
import HOG
import ChiSquare
import numpy as np
import sys
import K_means as K
import math
import time
import random
import copy

print("start")

img_infos = [] #the owner of each image`
ref_imgs = []
imgs = [] #input images
input_infos = []
threshold = 0.085
result1 = []
result2 = []
result3 = []
x = 0 #this variable is used in probability of each class
classes = {} #classes in reference images

# reading files
def reading_files(directory):
    img_infos = []  # the owner of each image`
    ref_imgs = []
    x = 0
    classes = {}  # classes in reference images
    for dirpath, dirs, files in os.walk('faces94'):
        for filename in fnmatch.filter(files, '*.jpg'):
            with open(os.path.join(dirpath, filename)):
                img_infos.append(str(filename.split('.')[0]))
                x += 1
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
    np.save('ref_img.npy', ref_imgs)
    np.save('img_infos.npy', img_infos)
    np.save('classes.npy', classes)
    return ref_imgs, img_infos, classes

#calculating distances between reference images
def distance_cal(ref_imgs, img_infos):
    distance_matrix = []
    for i in range (0, len(ref_imgs)):
        t1 = time.time()
        distance = []
        for j in range (0, len(ref_imgs)):
            if i == j:
                distance.append(0)
            else:
                distance.append(ChiSquare.distance_computation(ref_imgs[i], ref_imgs[j]))
        distance_matrix.append(distance)
        print(img_infos[i])
        print(time.time() - t1)
    np.savetxt("distances.csv", distance_matrix, delimiter=",")

# clustering reference images
def clustering(NC, ref_imgs, img_infos):
    clusters, centroids = K.KMeans_clustering(ref_imgs, NC, img_infos)
    np.save(clusters, 'clusters'+NC+'.npy')
    np.save(centroids, 'centroids' + NC + '.npy')

# loading saved files
ref_imgs = np.load("ref_img.npy")
img_infos = np.load("img_infos.npy")
classes = np.load("classes.npy").item()
distance_matrix = np.genfromtxt('distances.csv', delimiter=',')
test_files = np.genfromtxt('test_files.csv', delimiter=',') #loading the selected test files
test_files = test_files.astype(int)

# deleting the test files and their information from distance matrix and training set
for i in range(0,30):
    j = test_files[i]
    imgs.append(ref_imgs[j])
    input_infos.append(img_infos[j])
    ref_imgs = np.delete(ref_imgs, (j), axis=0)
    img_infos = np.delete(img_infos, (j), axis=0)
    distance_matrix = np.delete(distance_matrix, (j), axis=0)
    distance_matrix = np.delete(distance_matrix, (j), axis=1)

# loading clusters and centroids
clusters2, centroids2 =  np.load('clusters2.npy').item(), np.genfromtxt('centroids2.csv', delimiter=',')
centroids2 = centroids2.astype(int)
clusters3, centroids3 = np.load('clusters3.npy').item(), np.genfromtxt('centroids3.csv', delimiter=',')
centroids3 = centroids3.astype(int)
clusters11, centroids11 = K.KMeans_clustering(ref_imgs, 1,
                                              img_infos)
for max_iter in range(1, 200, 1): # We test the accuracy of each algorithm having different values for maximum iterations

    print(max_iter)

    #Testing the ML-ANN algorithm
    NC = 1  # number of clusters
    weights = np.zeros(NC)  # weights for each cluster
    CDistances = np.zeros(NC)  # dictances between clusters and input image
    distances = {}
    for i in range(0, NC):
        distances[str(i)] = []
    clusters, centroids = copy.deepcopy(clusters11), copy.deepcopy(centroids11)  # clustering reference images by K-Means algorithm
    # print(2)



    # print(3)
    clusters1, centroids1 = clusters, centroids
    average_time = 0
    numberOfImages = 0
    accuracy = 0
    for loop in range(0, 20):

        clusters, centroids = copy.deepcopy(clusters1), copy.deepcopy(centroids1)
        img = imgs[loop]
        queues = {}
        CDistances = np.zeros(NC)  # dictances between clusters and input image
        for i in range(0, NC):
            queues[str(i)] = []

        itr = 0  # iterations

        for i in range(0, NC):
            queues[str(i)].append(centroids[i])
            clusters[str(i)].remove(centroids[i])

        nearest = queues[str(0)][0]
        min_dist = ChiSquare.distance_computation(ref_imgs[queues[str(0)][0]], img)
        numberOfImages += 1
        distances[str(0)].append(min_dist)
        max_dist = min_dist
        CDistances[0] += min_dist

        for i in range(1, NC):
            distance = ChiSquare.distance_computation(ref_imgs[queues[str(i)][0]], img)
            numberOfImages += 1
            distances[str(i)].append(distance)
            CDistances[i] += distance
            if distance < min_dist:
                nearest = queues[str(i)][0]
                min_dist = distance

            if distance > max_dist:
                max_dist = distance

        weights = CDistances / (max_dist)
        weights = np.power(weights, -1)
        for i in range(0, len(weights)):
            weights[i] = math.ceil(weights[i])

        x = 0

        t = time.time()
        isEmpty = False

        while True:

            if min_dist <= threshold:
                average_time += time.time() - t
                if input_infos[loop] == img_infos[nearest]:
                    accuracy += 1
                break

            if itr > max_iter or isEmpty:
                average_time += time.time() - t
                if input_infos[loop] == img_infos[nearest]:
                    accuracy += 1
                break

            itr += 1
            argument = 0
            min = sys.maxsize

            isEmpty = True
            for j in range(0, len(clusters)):
                for w in range(0, math.ceil(weights[j]) * math.ceil(weights[j])):
                    argument = 0
                    min = sys.maxsize
                    for i in range(0, len(clusters[str(j)])):
                        likelihood = 0
                        for r_j in range(0, len(clusters)):
                            for r_i in range(0, len(queues[str(r_j)])):
                                fi = ((distances[str(r_j)][r_i] - distance_matrix[queues[str(r_j)][r_i]][
                                    clusters[str(j)][i]]) ** 2) / \
                                     distance_matrix[queues[str(r_j)][r_i]][clusters[str(j)][i]]
                                likelihood += fi
                        p = classes[img_infos[clusters[str(j)][i]]]
                        likelihood = likelihood - math.log(p)
                        if likelihood <= min:
                            min = likelihood
                            argument = i

                    if len(clusters[str(j)]) > 0:
                        distance = ChiSquare.distance_computation(ref_imgs[clusters[str(j)][argument]], img)
                        numberOfImages += 1
                        if distance < min_dist:
                            min_dist = distance
                            nearest = clusters[str(j)][argument]
                        CDistances[j] += distance
                        distances[str(j)].append(distance)
                        queues[str(j)].append(clusters[str(j)][argument])
                        del (clusters[str(j)][argument])
                        isEmpty = False

            max_dist = 0
            avg_distances = []
            for i in range(0, len(CDistances)):
                avg_distances.append(CDistances[i] / len(queues[str(i)]))
                if avg_distances[i] >= max_dist:
                    max_dist = avg_distances[i]
            # updating weight using average of distances from images in each queue and the input image
            weights = avg_distances / (max_dist)
            weights = np.power(weights, -1)
            for i in range(0, len(weights)):
                weights[i] = math.ceil(weights[i])
    print("1:")
    print("average time:")
    print(average_time)
    print("accuracy:")
    print(accuracy)
    print("number of distance computation:")
    print(numberOfImages)
    array = [max_iter, average_time, accuracy, numberOfImages]
    result1.append(array)
    #Testing the algorithm with 2 clusters
    NC = 2 #number of clusters
    distances = {}
    for i in range(0, NC):
        distances[str(i)] = []
    clusters, centroids = copy.deepcopy(clusters2), copy.deepcopy(centroids2)
    clusters1, centroids1 = clusters, centroids
    average_time = 0
    numberOfImages = 0
    accuracy = 0
    for loop in range(0, 20):

        clusters, centroids = copy.deepcopy(clusters1), copy.deepcopy(centroids1)
        img = imgs[loop]
        queues = {}
        CDistances = np.zeros(NC)  # dictances between clusters and input image
        for i in range(0, NC):
            queues[str(i)] = []

        itr = 0  # iterations

        for i in range(0, NC):
            queues[str(i)].append(centroids[i])
            clusters[str(i)].remove(centroids[i])

        nearest = queues[str(0)][0]
        min_dist = ChiSquare.distance_computation(ref_imgs[queues[str(0)][0]], img)
        numberOfImages += 1
        distances[str(0)].append(min_dist)
        max_dist = min_dist
        CDistances[0] += min_dist

        for i in range(1, NC):
            distance = ChiSquare.distance_computation(ref_imgs[queues[str(i)][0]], img)
            numberOfImages += 1
            distances[str(i)].append(distance)
            CDistances[i] += distance
            if distance < min_dist:
                nearest = queues[str(i)][0]
                min_dist = distance

            if distance > max_dist:
                max_dist = distance
        # calculating weight using average of distances from images in each queue and the input image
        weights = CDistances / (max_dist)
        weights = np.power(weights, -1)
        for i in range(0, len(weights)):
            weights[i] = math.ceil(weights[i])
        # print(weights)

        x = 0

        # print(4)

        t = time.time()
        isEmpty = False

        while True:

            if min_dist <= threshold:
                average_time += time.time() - t
                if input_infos[loop] == img_infos[nearest]:
                    accuracy += 1
                break

            if itr > max_iter or isEmpty:
                average_time += time.time() - t
                if input_infos[loop] == img_infos[nearest]:
                    accuracy += 1
                break

            itr += 1
            argument = 0
            min = sys.maxsize
            # print(7)
            isEmpty = True
            for j in range(0, len(clusters)):
                for w in range(0, math.ceil(weights[j]) * math.ceil(weights[j])):
                    argument = 0
                    min = sys.maxsize
                    for i in range(0, len(clusters[str(j)])):
                        # print(8)
                        likelihood = 0
                        for r_j in range(0, len(clusters)):
                            for r_i in range(0, len(queues[str(r_j)])):
                                fi = ((distances[str(r_j)][r_i] - distance_matrix[queues[str(r_j)][r_i]][
                                    clusters[str(j)][i]]) ** 2) / \
                                     distance_matrix[queues[str(r_j)][r_i]][clusters[str(j)][i]]
                                likelihood += fi
                        p = classes[img_infos[clusters[str(j)][i]]]
                        likelihood = likelihood - math.log(p)
                        if likelihood <= min:
                            min = likelihood
                            argument = i

                    if len(clusters[str(j)]) > 0:
                        distance = ChiSquare.distance_computation(ref_imgs[clusters[str(j)][argument]], img)
                        numberOfImages += 1
                        if distance < min_dist:
                            min_dist = distance
                            nearest = clusters[str(j)][argument]
                        CDistances[j] += distance
                        distances[str(j)].append(distance)
                        queues[str(j)].append(clusters[str(j)][argument])
                        del (clusters[str(j)][argument])
                        isEmpty = False

            max_dist = 0
            avg_distances = []
            for i in range(0, len(CDistances)):
                avg_distances.append(CDistances[i] / len(queues[str(i)]))
                if avg_distances[i] >= max_dist:
                    max_dist = avg_distances[i]
            # updating weight using average of distances from images in each queue and the input image
            weights = avg_distances / (max_dist)
            weights = np.power(weights, -1)
            for i in range(0, len(weights)):
                weights[i] = math.ceil(weights[i])

    print("2:")
    print("average time:")
    print(average_time)
    print("accuracy:")
    print(accuracy)
    print("number of distance computation:")
    print(numberOfImages)
    array = [max_iter, average_time, accuracy, numberOfImages]
    result2.append(array)

    #Testing the algorithm with 3 clusters
    NC = 3 #number of clusters
    distances = {}
    for i in range(0, NC):
        distances[str(i)] = []
    clusters, centroids = copy.deepcopy(clusters3), copy.deepcopy(centroids3)
    clusters1, centroids1 = clusters, centroids
    average_time = 0
    numberOfImages = 0
    accuracy = 0
    for loop in range(0, 20):

        clusters, centroids = copy.deepcopy(clusters1), copy.deepcopy(centroids1)
        img = imgs[loop]
        queues = {}
        CDistances = np.zeros(NC)  # dictances between clusters and input image
        for i in range(0, NC):
            queues[str(i)] = []

        itr = 0  # iterations

        for i in range(0, NC):
            queues[str(i)].append(centroids[i])
            clusters[str(i)].remove(centroids[i])

        nearest = queues[str(0)][0]
        min_dist = ChiSquare.distance_computation(ref_imgs[queues[str(0)][0]], img)
        numberOfImages += 1
        distances[str(0)].append(min_dist)
        max_dist = min_dist
        CDistances[0] += min_dist

        for i in range(1, NC):
            distance = ChiSquare.distance_computation(ref_imgs[queues[str(i)][0]], img)
            numberOfImages += 1
            distances[str(i)].append(distance)
            CDistances[i] += distance
            if distance < min_dist:
                nearest = queues[str(i)][0]
                min_dist = distance

            if distance > max_dist:
                max_dist = distance
        # calculating weight using average of distances from images in each queue and the input image
        weights = CDistances / (max_dist)
        weights = np.power(weights, -1)
        for i in range(0, len(weights)):
            weights[i] = math.ceil(weights[i])
        x = 0

        # print(4)

        t = time.time()
        isEmpty = False

        while True:

            if min_dist <= threshold:
                # print("distance:" + str(min_dist))
                # print(img_infos[nearest])
                # print(time.time() - t)
                average_time += time.time() - t
                # print(input_infos[loop])
                if input_infos[loop] == img_infos[nearest]:
                    accuracy += 1
                break

            if itr > max_iter or isEmpty:
                # print(queues)
                # print("The nearest class is: " + img_infos[nearest])
                # print(time.time() - t)
                average_time += time.time() - t
                if input_infos[loop] == img_infos[nearest]:
                    accuracy += 1
                break

            itr += 1
            argument = 0
            min = sys.maxsize
            isEmpty = True
            for j in range(0, len(clusters)):
                for w in range(0, math.ceil(weights[j]) * math.ceil(weights[j])):
                    argument = 0
                    min = sys.maxsize
                    for i in range(0, len(clusters[str(j)])):
                        # print(8)
                        likelihood = 0
                        for r_j in range(0, len(clusters)):
                            for r_i in range(0, len(queues[str(r_j)])):
                                fi = ((distances[str(r_j)][r_i] - distance_matrix[queues[str(r_j)][r_i]][
                                    clusters[str(j)][i]]) ** 2) / \
                                     distance_matrix[queues[str(r_j)][r_i]][clusters[str(j)][i]]
                                likelihood += fi
                        p = classes[img_infos[clusters[str(j)][i]]]
                        likelihood = likelihood - math.log(p)
                        if likelihood <= min:
                            min = likelihood
                            argument = i

                    if len(clusters[str(j)]) > 0:
                        distance = ChiSquare.distance_computation(ref_imgs[clusters[str(j)][argument]], img)
                        numberOfImages += 1
                        if distance < min_dist:
                            min_dist = distance
                            nearest = clusters[str(j)][argument]
                        CDistances[j] += distance
                        distances[str(j)].append(distance)
                        queues[str(j)].append(clusters[str(j)][argument])
                        del (clusters[str(j)][argument])
                        isEmpty = False

            max_dist = 0
            avg_distances = []
            for i in range(0, len(CDistances)):
                avg_distances.append(CDistances[i] / len(queues[str(i)]))
                if avg_distances[i] >= max_dist:
                    max_dist = avg_distances[i]
            # updating weight using average of distances from images in each queue and the input image
            weights = avg_distances / (max_dist)
            weights = np.power(weights, -1)
            for i in range(0, len(weights)):
                weights[i] = math.ceil(weights[i])

    print("3:")
    print("average time:")
    print(average_time)
    print("accuracy:")
    print(accuracy)
    print("number of distance computation:")
    print(numberOfImages)
    print("--------------------------------------------------------")
    print()
    print()
    array = [max_iter, average_time, accuracy, numberOfImages]
    result3.append(array)
#     np.savetxt("result1.csv", result1, delimiter=",")
#     np.savetxt("result2.csv", result2, delimiter=",")
#     np.savetxt("result3.csv", result3, delimiter=",")
#
# np.savetxt("result1.csv", result1, delimiter=",")
# np.savetxt("result2.csv", result2, delimiter=",")
# np.savetxt("result3.csv", result3, delimiter=",")

