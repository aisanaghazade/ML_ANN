import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = './test.JPG'

img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# gray = np.float32(gray)




# dst = cv2.cornerHarris(gray,4,5,0)
#
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
#
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
#
# dst = np.uint8(dst)
#
# # find centroids
# ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
#
# # define the criteria to stop and refine the corners
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
#
# # Now draw them
# res = np.hstack((centroids,corners))
# res = np.int0(res)
# img[res[:,1],res[:,0]]=[0,0,255]
# img[res[:,3],res[:,2]] = [0,255,0]
#
# cv2.imwrite('subpixel5.png',img)

# corners = cv2.goodFeaturesToTrack(gray,100,0.001,10)
# corners = np.int0(corners)
#
# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),3,255,-1)
#
# plt.imshow(img),plt.show()

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('sift_keypoints.jpg',img)




# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#
#
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()