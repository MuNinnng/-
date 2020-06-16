import cv2
import matplotlib.pylab as plt
import numpy as np
import pylab

assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3, 'OpenCV version 3 or newer required.'
'''
DIM=(2592, 1944)
K=np.array([[560.7199488745102, 0.0, 1237.3184109068156], [0.0, 559.69820192407, 1010.1282871767485], [0.0, 0.0, 1.0]])
D=np.array([[0.08878506996902752], [-0.03941956482288721], [0.03720614272658149], [-0.028198419201761714]])
((2592, 1944), array([[5.60719949e+02, 0.00000000e+00, 1.23731841e+03],
       [0.00000000e+00, 5.59698202e+02, 1.01012829e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), array([[ 0.08878507],
       [-0.03941956],
       [ 0.03720614],
       [-0.02819842]]))

'''
K = np.array([[560.7199488745102, 0.0, 1237.3184109068156], [0.0, 559.69820192407, 1010.1282871767485], [0.0, 0.0, 1.0]])


# zero distortion coefficients work well for this image
# D = np.array([0., 0., 0., 0.])

D=np.array([[0.08878506996902752], [-0.03941956482288721], [0.03720614272658149], [-0.028198419201761714]])

# use Knew to scale the output
Knew = K.copy()
Knew[(0,1), (0,1)] = 1.0 * Knew[(0,1), (0,1)]


img = cv2.imread('F:/photo_2/image_num_0_today.jpg')
plt.figure('original image')
plt.imshow(img[::-1])
pylab.show()
img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
cv2.imwrite('right_60_undistorted.jpg', img_undistorted)
plt.figure('undistored image')
plt.imshow(img_undistorted[::-1])
pylab.show()
