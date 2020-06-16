import cv2
import matplotlib.pylab as plt
import numpy as np
import pylab
import sys
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
DIM=(2592, 1944)
K=np.array([[560.7199488745102, 0.0, 1237.3184109068156], [0.0, 559.69820192407, 1010.1282871767485], [0.0, 0.0, 1.0]])
D=np.array([[0.08878506996902752], [-0.03941956482288721], [0.03720614272658149], [-0.028198419201761714]])
def undistort(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    img_1=cv2.imshow("undistorted", undistorted_img)
    plt.figure('original image')
    plt.imshow(img_1[::-1])
    pylab.show()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
