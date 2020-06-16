import cv2
import sift as sift
import pylab

img = cv2.imread('E:/Plots/myplot4.png')
gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
img1=cv2.drawKeypoints(gray, kps, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT_Algorithm', img1)
pylab.show()

surf = cv2.xfeatures2d.SURF_create()
(kps2, descs2) = surf.detectAndCompute(gray, None)
img2=cv2.drawKeypoints(gray, kps2, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SURF_Algorithm', img2)
pylab.show()

fast = cv2.FastFeatureDetector_create()
kps3 = fast.detect(gray, None)
img3=cv2.drawKeypoints(gray, kps3, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('FAST_Algorithm', img3)
pylab.show()

orb = cv2.ORB_create()
kps4 = orb.detect(gray, None)
(kps4, des4) = orb.compute(gray, kps4)
img4=cv2.drawKeypoints(gray, kps4, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('ORB_Algorithm', img4)
pylab.show()