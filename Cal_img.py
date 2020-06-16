import numpy as np

from calibration.fisheye.Calibration import img_undistorted, img


def cal_img(img):
    num_coor_row = []
    num_coor_col = []
    # 记录每行有多少个黑色像素
    for i in range(img.shape[0]):
        num_row = 0
        for j in range(img.shape[1]):
            if sum(img[i][j]) == 0:
                # print([i, j])
                num_row = num_row + 1
        num_coor_row.append(num_row)
    # 记录每列有多少个黑色像素
    for k in range(img.shape[1]):
        num_col = 0
        for l in range(img.shape[0]):
            if sum(img[l][k]) == 0:
                # print([l, k])
                num_col = num_col + 1
        num_coor_col.append(num_col)
    return num_coor_row, num_coor_col


def multi_index(alist, f):    # 返回list中多个元素的索引
    return [i for (i,v) in enumerate(alist)if v == f]


# 去除图像黑色边缘
num_coor_row, num_coor_col = cal_img(img_undistorted)
edge_index_row = multi_index(num_coor_row, img.shape[1])
edge_index_col = multi_index(num_coor_col, img.shape[0])
img_cropped = np.delete(img_undistorted, edge_index_row, axis=0)
img_cropped = np.delete(img_cropped, edge_index_col, axis=1)
plt.figure('cropped image')
plt.imshow(img_cropped[..., ::-1])


def crop_img(img_cropped):
    # 截取上下感兴趣区域
    top_roi = img_cropped[:int(img_cropped.shape[0] / 2),
                          int(img_cropped.shape[1] / 2) - 50:int(img_cropped.shape[1] / 2) + 50, :]
    bottom_roi = img_cropped[int(img_cropped.shape[0] / 2) + 1:,
                             int(img_cropped.shape[1] / 2) - 50:int(img_cropped.shape[1] / 2) + 50, :]
    # 统计黑色像素个数
    _, top_num_coor_row = cal_img(top_roi)
    _, bottom_num_coor_row = cal_img(bottom_roi)
    # 截取左右感兴趣区域
    left_roi = img_cropped[int(img_cropped.shape[0] / 2) - 50:int(img_cropped.shape[0] / 2) + 50,
                           :int(img_cropped.shape[1] / 2), :]
    right_roi = img_cropped[int(img_cropped.shape[0] / 2) - 50:int(img_cropped.shape[0] / 2) + 50,
                            int(img_cropped.shape[1] / 2) + 1:, :]
    # 统计黑色像素个数
    left_num_coor_col, _ = cal_img(left_roi)
    right_num_coor_col, _ = cal_img(right_roi)
    # plt.figure('roi')
    # plt.imshow(top_roi)
    '''
    print(top_num_coor_row)
    print(bottom_num_coor_row)
    print(left_num_coor_col)
    print(right_num_coor_col)
    '''
    # 确定黑色像素最多的行和列的索引
    top_max_row = max(top_num_coor_row)
    top_max_row_index = multi_index(top_num_coor_row, top_max_row)
    bottom_max_row = max(bottom_num_coor_row)
    bottom_max_row_index = multi_index(bottom_num_coor_row, bottom_max_row)
    left_max_col = max(left_num_coor_col)
    left_max_col_index = multi_index(left_num_coor_col, left_max_col)
    right_max_col = max(right_num_coor_col)
    right_max_col_index = multi_index(right_num_coor_col, right_max_col)
    '''
    print(top_max_row_index)
    print(bottom_max_row_index)
    print(left_max_col_index)
    print(right_max_col_index)
    '''
    # 取索引的中位数
    top_col_index = int(np.median(top_max_row_index))
    bottom_col_index = int(np.median(bottom_max_row_index))
    left_row_index = int(np.median(left_max_col_index))
    right_row_index = int(np.median(right_max_col_index))
    print(top_col_index,bottom_col_index,left_row_index,right_row_index)
    # 取中位数对应的列
    top_col = top_roi[:, top_col_index, :]
    bottom_col = bottom_roi[:, bottom_col_index, :]
    # 该列第一个不为零的像素为校正后图像内切矩形的上切点
    for top_c in range(top_roi.shape[0]):
        if sum(top_col[top_c]) != 0:
            top = top_c
            break
    # 该列第一个为零的像素为校正后图像内切矩形的下切点
    for bottom_c in range(bottom_roi.shape[0]):
        if sum(bottom_col[bottom_c+1]) == 0:
            bottom = bottom_c
            break
    # 取中位数对应的行
    left_row = left_roi[left_row_index, :, :]
    right_row = right_roi[right_row_index, :, :]
    # 该行第一个不为零的像素为校正后图像内切矩形的上切点
    for left_r in range(left_roi.shape[1]):
        if sum(left_row[left_r]) != 0:
            left = left_r
            break
    # 该行第一个不为零的像素为校正后图像内切矩形的上切点
    for right_r in range(int(img_cropped.shape[1] / 2)):
        if sum(right_row[right_r+1]) == 0:
            right = right_r
            break
    # 计算上下左右切点的真实索引
    top = top
    bottom = bottom + top_roi.shape[0]
    left = left
    right = right + left_roi.shape[1]
    return top, bottom, left, right


top_f,bottom_f,left_f,right_f = crop_img(img_cropped)


print(top_f,bottom_f,left_f,right_f)
img_cropped_f = img_cropped[top_f:bottom_f, left_f:right_f, :]
plt.figure('cropped image')


plt.figure('final result')
plt.imshow(img_cropped_f[...,::-1])
plt.show()
cv2.imwrite('final_result.jpg', img_cropped_f)
