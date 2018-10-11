import cv2
import numpy as np

img1 = cv2.imread('test.JPG')

img2 = cv2.imread('test2.JPG')

h1 = []
h2 = []

# print(str(len(img1))+"  "+str(len(img1[0]))+"  "+str(len(img2))+"  "+str(len(img2[0])))
for k in range(0,30):
    h1.append([0,0,0])
    h2.append([0,0,0])
    for i in range(0,1332):
        for j in range(0,999):
            if(img1[i][j][0] == k):
                h1[k][0]+=1;
            if (img1[i][j][1] == k):
                h1[k][1] += 1;
            if (img1[i][j][2] == k):
                h1[k][2] += 1;
            if(img2[i][j][0] == k):
                h2[k][0] += 1;
            if (img2[i][j][1] == k):
                h2[k][1] += 1;
            if (img2[i][j][2] == k):
                h2[k][2] += 1;

    h1[k][0] = h1[k][0] / (len(img1) * len(img1[0]))
    h1[k][1] = h1[k][1] / (len(img1) * len(img1[0]))
    h1[k][2] = h1[k][2] / (len(img1) * len(img1[0]))
    h2[k][0] = h2[k][0] / (len(img2) * len(img2[0]))
    h2[k][1] = h2[k][1] / (len(img2) * len(img2[0]))
    h2[k][2] = h2[k][2] / (len(img2) * len(img2[0]))

print(h1)
print(h2)

# KL distance measure
ro1 = 0
ro2 = 0
ro3 = 0
for k in range(0,30):
    ro1 += h1[k][0] * np.log(h1[k][0]/h2[k][0])
    ro2 += h1[k][1] * np.log(h1[k][1]/h2[k][1])
    ro3 += h1[k][2] * np.log(h1[k][2]/h2[k][2])

print(ro1)
print(ro2)
print(ro3)

normalizedImg = np.zeros((800, 800))
normalizedImg = cv2.normalize(img1,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('dst_rt', normalizedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()