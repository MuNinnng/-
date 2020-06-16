import cv2
import numpy as np
import os
import glob
import matplotlib.pylab as plt


def get_K_and_D(checkerboard, imgsPath):
    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    #fisheye::CALIB_USE_INTRINSIC_GUESS ：cameraMatrix包含有效的初始值的fx, fy, cx, cy进一步优化。否则，(cx, cy)初始设置为图像中心(使用imageSize)，并以最小二乘方式计算焦距。
    #fisheye::CALIB_RECOMPUTE_EXTRINSIC：在每次内部优化迭代之后，外部的将被重新计算。
    #fisheye::CALIB_CHECK_COND：这些函数将检查条件号的有效性。
    #fisheye::CALIB_FIX_SKEW：歪斜系数(alpha)被设置为零并保持零。
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    # 用数组的形式来保存每一幅棋盘格板中所有内角点的三维坐标。
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    #print(objp)
    _img_shape = None
    objpoints = []
    imgpoints = []
    images = glob.glob(imgsPath + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        plt.figure('original image')
        plt.imshow(img[::-1])
        if _img_shape == None:
            '"取图像高/宽"'
            _img_shape = img.shape[:2]
        else:
            '"所有图像必须同高同宽"'
            assert _img_shape == img.shape[:2]

        "img转化为灰度图，ret判断是否有图，提取角点"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        for c in corners:
            plt.plot(c[0][0], c[0][1], 'r*')
        # plt.show()
        if ret == True:
            objpoints.append(objp)
            'cv2.cornerSubPix()"为了提高标定精度，需要在初步提取的角点信息上进一步提取亚像素信息，降低相机标定偏差"'
            'cornerSubPix(image, corners, winSize, zeroZone, criteria)'
            '第一个参数image，输入图像的像素矩阵，最好是8位灰度图像，检测效率更高；'
            '第二个参数corners，初始的角点坐标向量，同时作为亚像素坐标位置的输出，所以需要是浮点型数据；'
            '第三个参数winSize，大小为搜索窗口的一半；'
            '第四个参数zeroZone，死区的一半尺寸，死区为不对搜索区的中央位置做求和运算的区域。它是用来避免自相关矩阵出现某些可能的奇异性。当值为（-1，-1）时表示没有死区；'
            '第五个参数criteria，定义求角点的迭代过程的终止条件，可以为迭代次数和角点精度两者的组合'
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    'tvecs表示平移，而rvecs表示旋转'
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    rms, a, b, c, d = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    #print("旋转"+str(rvecs))
    #print("平移"+str(tvecs))
    print("rms="+str(rms))
    print("a=" + str(a))
    print("b=" + str(b))
    print("c=" + str(c))
    print("d=" + str(d))
    return DIM, K, D


# 计算内参和矫正系数
'''
# checkerboard： 棋盘格的格点数目
# imgsPath: 存放鱼眼图片的路径
'''
print(get_K_and_D((6, 9), r'C:\Users\DELL\Desktop\photo_1'))


