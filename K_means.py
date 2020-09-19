import ChiSquare
import random
from sklearn.cluster import KMeans

# this file has to function for clustering. we use the second one.

def KMeans_clustering1(train, NC, iter, info):#NC = Number of Clusters, i = max_iteration
    iteration = 0
    centroids = random.sample(range(0, len(train)), NC)
    for i in range(0, len(centroids)):
        print(info[centroids[i]])
    clusters = {}

    for i in range(0, NC):
        clusters[str(i)] = []

    for i in range(0, len(train)):
        cent = 0 #centroid
        min_dist = ChiSquare.distance_computation(train[centroids[0]], train[i])
        for j in range(1, len(centroids)):
            distance = ChiSquare.distance_computation(train[centroids[j]], train[i])
            if distance < min_dist:
                min_dist = distance
                cent = j
        clusters[str(cent)].append(i)

    for i in range(0, len(centroids)):
        sum = train[clusters[str(i)][0]]
        for j in range(1, len(clusters[str(i)])):
            sum += train[clusters[str(i)][j]]
        mean = sum / len(clusters[str(i)])
        min_dist = ChiSquare.distance_computation(train[clusters[str(i)][0]], mean)
        index = 0
        for j in range(1, len(clusters[str(i)])):
            distance = ChiSquare.distance_computation(train[clusters[str(i)][j]], mean)
            if distance < min_dist:
                min_dist = distance
                index = j
        centroids[i] = index
    prev_cost = 0
    for i in range(0, len(centroids)):
        for j in range(0, len(clusters[str(i)])):
            for k in range(0, len(clusters[str(i)])):
                prev_cost += ChiSquare.distance_computation(train[clusters[str(i)][j]], train[clusters[str(i)][k]])

    while True:
        iteration += 1
        print(iteration)
        if iteration > iter:
            print("lanat behet")
            break

        for i in range(0, NC):
            clusters[str(i)] = []

        for i in range(0, len(train)):
            cent = 0  # centroid
            min_dist = ChiSquare.distance_computation(train[centroids[0]], train[i])
            for j in range(1, len(centroids)):
                distance = ChiSquare.distance_computation(train[centroids[j]], train[i])
                if centroids[j] == i:
                    print(info[centroids[j]]+" : "+info[i])
                    min_dist = 0
                    cent = j
                    break
                if distance < min_dist:
                    min_dist = distance
                    cent = j
            clusters[str(cent)].append(i)

        for i in range(0, len(centroids)):
            # if len(clusters[str(i)]) == 0:
            #     continue
            print(str(len(train))+":"+str(clusters[str(i)][0]))
            sum = train[clusters[str(i)][0]]
            for j in range(1, len(clusters[str(i)])):
                sum += train[clusters[str(i)][j]]
            mean = sum / len(clusters[str(i)])
            min_dist = ChiSquare.distance_computation(train[clusters[str(i)][0]], mean)
            index = 0
            for j in range(1, len(clusters[str(i)])):
                distance = ChiSquare.distance_computation(train[clusters[str(i)][j]], mean)
                if distance < min_dist:
                    min_dist = distance
                    index = j
            centroids[i] = index
        current_cost = 0
        for i in range(0, len(centroids)):
            for j in range(0, len(clusters[str(i)])):
                for k in range(0, len(clusters[str(i)])):
                    current_cost += ChiSquare.distance_computation(train[clusters[str(i)][j]], train[clusters[str(i)][k]])


        prev_cost = current_cost



    for i in range(0, len(centroids)):
        for j in range(0, len(clusters[str(i)])):
            print(info[centroids[i]] + " : " + info[clusters[str(i)][j]])


def KMeans_clustering(train, NC, info):
    clusters = {}
    centroids = []

    for i in range(0, NC):
        clusters[str(i)] = []
    train_set = []
    for i in range(0, len(train)):
        x = []
        for j in range(0, len(train[0])):
            for k in range(0, len(train[0][0])):
                for l in range(0, len(train[0][0][0])):
                    x.append(train[i][j][k][l])
        train_set.append(x)

    KM = KMeans(NC)
    KM.fit(train_set)
    output = KM.predict(train_set)
    for i in range(0, len(output)):
        print(str(info[i])+" : "+str(output[i]))
        clusters[str(output[i])].append(i)
    for i in range(0, NC):
        sum = 0
        for j in range(1, len(clusters[str(i)])):
            sum += train[clusters[str(i)][j]]
        mean = sum / len(clusters[str(i)])
        min_dist = ChiSquare.distance_computation(train[clusters[str(i)][0]], mean)
        index = 0
        for j in range(1, len(clusters[str(i)])):
            distance = ChiSquare.distance_computation(train[clusters[str(i)][j]], mean)
            if distance < min_dist:
                min_dist = distance
                index = clusters[str(i)][j]
        centroids.append(index)
        print(centroids[i])
        print(clusters[str(i)])
    return clusters, centroids

