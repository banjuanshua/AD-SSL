import os
import cv2
import time
import torch
import numba
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# KITTI image 02 std

# K = [
#     [9.597910e+02, 0, 6.960217e+02],
#     [0, 9.569251e+02, 2.241806e+02],
#     [0, 0, 1.0]
# ]

K = [
    [9.597910e+02, 0, 500],
    [0, 9.569251e+02, 2.241806e+02],
    [0, 0, 1.0]
]


R = [
    [9.999758e-01, -5.267463e-03, -4.552439e-03],
    [5.251945e-03, 9.999804e-01, -3.413835e-03],
    [4.570332e-03, 3.389843e-03, 9.999838e-01]
]

T = [5.956621e-02, 2.900141e-04, 2.577209e-03]

def read_img_depth(img, disp_np=None):
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            # print('point depth', x, y, 1/disp_np[x, y])

            # depth = 0.1 + (100-0.1) * (1/disp_np[x,y])
            # print('point depth', x, y, disp_np[x][y])
            print('point depth', x, y, img[y][x])

            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    while (1):
        cv2.imshow("image", img)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def windows_serach(img_lane_single):
    out_img = img_lane_single.copy()
    lane_single = img_lane_single[:,:,0]
    # print(out_img.shape)
    # print(lane_single.shape)
    # sss

    margin = 30
    nwindows = 50
    # Set minimum number of pixels found to recenter window
    h_thresh =  500
    roi_thresh = 50

    window_height = lane_single.shape[0] // nwindows

    histogram = np.sum(lane_single, axis=0)
    # vertical_histogram = np.sum(lane_single[lane_single.shape[0] // 2:, :], axis=0)
    # horizontal_histogram = np.sum(lane_single[lane_single.shape])
    index = np.where(histogram>h_thresh)[0]
    x_current = index[0]

    nonzero = lane_single.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

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
            x_current = int(np.mean(nonzerox[roi_inds]))
            x_mean = int(np.mean(nonzerox[roi_inds]))
            y_mean = int(np.mean(nonzeroy[roi_inds]))

            cv2.circle(out_img, (x_mean, y_mean), 1, (255, 0, 0), thickness=4)
            cv2.rectangle(out_img,(win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            lane_points_x.append(x_mean)
            lane_points_y.append(y_mean)

    return out_img, lane_points_x[:-2], lane_points_y[:-2]

def fit_poly2d(x_arr, y_arr, coeff=2, img_shape=None):
    if img_shape != None:
        y_len = img_shape[0]
    else:
        y_len = y_arr[0]
    poly = np.polyfit(y_arr, x_arr, coeff)

    # y_len/2 只输出到半高，超过高度的一半时不输出车道线
    y_arr = np.linspace(int(y_len/2), y_len, y_len)
    x_arr = poly[0]*y_arr**2 + poly[1]*y_arr + poly[2]
    return poly, x_arr, y_arr


def pointsTo3d_single_test(x_arr, y_arr, img_depth):
    # print(img_depth.shape)
    x, y = x_arr[0], y_arr[0]
    z = img_depth[y][x]
    p2d = np.array([x, y, 1], np.float32)
    p2d *= z

    print(x, y, z)
    invert_K = np.linalg.pinv(K)
    p3d = np.dot(invert_K, p2d)
    # print(p3d * z)
    print(p3d)






def process_lane(img_raw, img_lane_single, disp_np):
    img_depth = 1/disp_np * 100

    # lps: lane_points
    out_img, lps_x, lps_y = windows_serach(img_lane_single)
    lane_poly, poly_x_arr, poly_y_arr = fit_poly2d(lps_x, lps_y, img_shape=img_lane_single.shape)

    points3d = image2sapce(lps_x, lps_y, img_depth)
    # read_img_depth(lane_depth)

    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(img_raw)

    plt.subplot(3, 1, 2)
    plt.imshow(out_img)

    plt.subplot(3, 1, 3)
    plt.plot(poly_x_arr, poly_y_arr, color='r')
    plt.imshow(img_raw)

    x = [i[0] for i in points3d][:-6]
    # y = [i[1] for i in points3d][:-6]
    z = [i[2] for i in points3d][:-6]
    y = [0 for i in range(len(points3d))][:-6]

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(0, 0, 0, color='r', label='lane line')
    # ax.plot(x, z, y, color='b')

    plt.show()


def run():
    img_raw = cv2.imread('a.png')
    img_lane = cv2.imread('predict.jpg')
    disp_np = np.load('a_disp.npy').squeeze()
    # img_lane_single = cv2.imread('lane_single.jpg')


    lane_raw = np.load('lane.npy')
    lane_single = lane_raw[:,1,:,:].squeeze() * 255
    lane_single[lane_single<10] = 0
    # lane_single = cv2.blur(lane_single, (9, 9))
    img_lane_single = np.dstack((lane_single, lane_single, lane_single))
    # cv2.imshow('sdfds', img_lane_single)
    # cv2.waitKey(0)

    img_raw = cv2.resize(img_raw, (1024, 320))
    img_lane = cv2.resize(img_lane, (1024, 320))
    img_lane_single = cv2.resize(img_lane_single, (1024, 320))


    # print(img_lane_single.shape)
    # print(lane_raw.shape)
    #
    # sss

    process_lane(img_raw, img_lane_single, disp_np)


    # read_img_depth(img_lane, disp_np)
    # read_img_depth(lane_depth, 0)

    # cv2.imshow('raw', img_raw)
    # cv2.imshow('lane', img_lane)
    # cv2.imshow('disp', img_disp)
    # cv2.imshow('lane_single', lane_depth)
    # cv2.waitKey(0)




if __name__ == '__main__':
    run()