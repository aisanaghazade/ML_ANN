import numpy as np
import cv2
import ChiSquare

pixels1 = cv2.imread('faces94/female/9336923/9336923.1.jpg')
pixels2 = cv2.imread('faces94/female/9338535/9338535.1.jpg')
# max = 255
# min = 0
# histo = np.zeros(max + 1)
# height = len(pixels)
# width = len(pixels[0])
# # for i in range(0, height):
# #     for j in range(0, width):
# #         histo[pixels[i][j]]
#
# print(pixels[0][0])

ChiSquare.distance_computation(pixels1, pixels2)
