import glob
import os
import fnmatch
import cv2
import HOG
import HTPNN
import numpy as np

img_infos = {}
ref_imgs = []


x = 0
for dirpath, dirs, files in os.walk('faces94'):
    for filename in fnmatch.filter(files, '*.jpg'):
        with open(os.path.join(dirpath, filename)):
            img_infos[x] = filename.split('.')[0]
            x += 1
            # ref_imgs.append(cv2.imread(os.path.join(dirpath, filename)))
            ref_imgs.append(HOG.HOG(os.path.join(dirpath, filename)))


distance_matrix = []
for i in range (0, len(ref_imgs)):
    distance = []
    for j in range (0, len(ref_imgs)):
        if i == j:
            distance.append(0)
        else:
            distance.append(HTPNN.htpnn(ref_imgs[i],ref_imgs[j]))
    distance_matrix.append(distance)

np.savetxt("distances.csv", distance_matrix, delimiter=",")

distance_matrix = np.
