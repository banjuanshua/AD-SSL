import re
import os
import cv2
import time
import inspect
import torch
import numba
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from atutils.geometricu import *

# KITTI image 02 std


# KiTTi std
K = [
    [9.597910e+02, 0, 6.960217e+02],
    [0, 9.569251e+02, 2.241806e+02],
    [0, 0, 1.0]
]

K = [
    [9.597910e+02, 0, 500],
    [0, 9.569251e+02, 2.241806e+02],
    [0, 0, 1.0]
]

K = [
    [721, 0, 500],
    [0, 721, 147],
    [0, 0, 1.0]
]


R = [
    [9.999758e-01, -5.267463e-03, -4.552439e-03],
    [5.251945e-03, 9.999804e-01, -3.413835e-03],
    [4.570332e-03, 3.389843e-03, 9.999838e-01]
]

T = [5.956621e-02, 2.900141e-04, 2.577209e-03]



def windows_serach(img_lane_single):
    # todo 网络修改后，传入lane_signle_raw

    out_img = img_lane_single.copy()
    lane_single = img_lane_single[:,:,0]

    # shape_check(out_img, lane_single)

    margin = 50
    nwindows = 50
    # Set minimum number of pixels found to recenter window
    h_thresh = 500
    roi_thresh = 50

    window_height = lane_single.shape[0] // nwindows

    histogram = np.sum(lane_single, axis=0)
    # todo 较大的曲线应使用两个方向的直方图
    # vertical_histogram = np.sum(lane_single[lane_single.shape[0] // 2:, :], axis=0)
    # horizontal_histogram = np.sum(lane_single[lane_single.shape])
    index = np.where(histogram>h_thresh)[0]
    x_current = index[0]

    nonzero = lane_single.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if nonzerox[-1] > nonzerox[0]:
        nonzeroy = nonzeroy[::-1]
        nonzerox = nonzerox[::-1]
        x_current = index[-1]

    lane_points_x = []
    lane_points_y = []
    for window in range(nwindows):
        # 定义据矩形ROI
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        win_y_low = lane_single.shape[0] - (window + 1) * window_height
        win_y_high = lane_single.shape[0] - window * window_height

        # 找到矩形ROI中点的坐标
        # roi_inds为nonzerox/y的index，nonzeros中即为坐标
        roi_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        if len(roi_inds) > roi_thresh:
            x_mean = int(np.mean(nonzerox[roi_inds]))
            y_mean = int(np.mean(nonzeroy[roi_inds]))

            cv2.circle(out_img, (x_mean, y_mean), 1, (255, 0, 0), thickness=4)
            cv2.rectangle(out_img,(win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            lane_points_x.append(x_mean)
            lane_points_y.append(y_mean)

            x_current = x_mean

    return out_img, lane_points_x, lane_points_y



def display2d(img_raw, lane_poly2d, img_lane_method):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(img_raw)

    plt.subplot(3, 1, 2)
    # plt.imshow(img_lane_method)

    plt.subplot(3, 1, 3)
    y_len = img_raw.shape[0]
    for poly in lane_poly2d:
        # y_len/2 只输出到半高，超过高度的一半时不输出车道线
        y_arr = np.linspace(int(y_len / 2), y_len, y_len)
        x_arr = poly[0] * y_arr ** 2 + poly[1] * y_arr + poly[2]
        plt.plot(x_arr, y_arr, color='r')

    plt.imshow(img_raw)



def display3d(lane_poly3d):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(0, 0, 0, color='r', label='lane line')

    z_arr = np.array([i for i in range(5, 100)])
    y_arr = np.array([0 for i in range(5, 100)])
    for poly in lane_poly3d:
        x_arr = poly[0]*z_arr**2 + poly[1]*z_arr + poly[2]
        ax.plot(x_arr, z_arr, y_arr, color='b')


def process_lane(lane_raw, img_depth):
    lane_poly2d = []
    lane_poly3d = []
    img_lane_method = None
    lane_nums = lane_raw.shape[0]

    for i in range(lane_nums):
        lane_single = lane_raw[i].copy() * 255
        # filter
        lane_single[lane_single<10] = 0
        img_lane_single = np.dstack((lane_single, lane_single, lane_single))

        # 改变lane_raw的shape
        img_lane_single = cv2.resize(img_lane_single, (1024, 320))

        # lps: lane points
        out_img, lps_x, lps_y = windows_serach(img_lane_single)
        if img_lane_method is None:
            img_lane_method = out_img
        else:
            img_lane_method = cv2.addWeighted(out_img, 1, img_lane_method, 1, 0)

        lane_poly2d_tmp = fit_poly2d(lps_x, lps_y)
        points3d = image2sapce(lps_x, lps_y, img_depth, K)
        lane_poly3d_tmp = fit_poly3d(points3d)

        lane_poly2d.append(lane_poly2d_tmp)
        lane_poly3d.append(lane_poly3d_tmp)

    return lane_poly2d, lane_poly3d, img_lane_method


def shape_check(*args):
    def varname(p):
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
            if m:
                return m.group(1)

    for x in args:
        if type(x) == list:
            print(len(x))
        else:
            print(x.shape)
    sss

def run():
    img_raw = cv2.imread('tstdata/a.png')
    disp_np = np.load('tstdata/a_disp.npy').squeeze()
    lane_raw = np.load('tstdata/lane.npy').squeeze()
    lane_raw = lane_raw[1:4,:,:]

    img_raw = cv2.resize(img_raw, (1024, 320))
    img_depth = 1 / disp_np * 100

    # shape_check(img_raw, disp_np, lane_raw)

    lane_poly2d, lane_poly3d, img_lane_method = process_lane(lane_raw, img_depth)


    display2d(img_raw, lane_poly2d, img_lane_method)
    display3d(lane_poly3d)
    plt.show()



if __name__ == '__main__':
    run()