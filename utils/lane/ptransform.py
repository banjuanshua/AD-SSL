import cv2
import sys
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt


import atutils.imgu as ImgUt

dect = [[450, 371],
        [775, 371],
        [450, 184],
        [775, 184]]

src = [[450, 371],
       [775, 371],
       [598, 184],
       [614, 184]]

dst = [[150, 900],
       [250, 900],
       [150, 50],
       [250, 50]]

def perpective_trans():
    global src
    global dst
    src = np.float32(src)
    dst = np.float32(dst)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def draw_dect_area(img):
    # roi
    cv2.line(img, tuple(dect[0]), tuple(dect[1]), (0,255,0), 2)
    cv2.line(img, tuple(dect[0]), tuple(dect[2]), (0, 255, 0), 2)
    cv2.line(img, tuple(dect[3]), tuple(dect[1]), (0, 255, 0), 2)
    cv2.line(img, tuple(dect[3]), tuple(dect[2]), (0, 255, 0), 2)


    # lane
    cv2.line(img, tuple(dect[0]), tuple(src[2]), (255,0,0), 2)
    cv2.line(img, tuple(dect[1]), tuple(src[3]), (255, 0, 0), 2)

    return img


def draw_lane(img, llp, rlp):
    cv2.line(img, tuple(dst[0]), tuple(dst[2]), (255, 0, 0), 2)
    cv2.line(img, tuple(dst[1]), tuple(dst[3]), (255, 0, 0), 2)
    return img


def get_lane_poly():
    n_deg = 1

    x = [dst[0][0], dst[2][0]]
    y = [dst[0][1], dst[2][1]]
    llp = np.poly1d(np.polyfit(x, y, n_deg))

    x = [dst[1][0], dst[3][0]]
    y = [dst[1][1], dst[3][1]]
    rlp = np.poly1d(np.polyfit(x, y, n_deg))

    print(llp.coeffs)
    print(rlp.coeffs)

    return llp, rlp



def do_bev():
    videp_path = 'output.mp4'
    cap = cv2.VideoCapture(videp_path)

    M, M_inv = perpective_trans()
    bev_size = (1000, 400)
    bev = np.zeros(bev_size)

    llp, rlp = get_lane_poly()


    while (cap.isOpened()):
        ret, img = cap.read()
        img = draw_dect_area(img)

        # im = Image.fromarray(img)
        # im = im.convert('L')
        # warped = cv2.warpPerspective(bev, M, bev_size, flags=cv2.INTER_LINEAR)
        # ret, warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)


        bev = draw_lane(bev, llp, rlp)


        cv2.imshow('frame', img)
        # cv2.imshow('warped', warped)
        cv2.imshow('bev', bev)


        # plt.clf()
        # plt.title("lanes and path")
        # plt.imshow(img)
        # plt.scatter(range(0,192), range(0, 192), c="b")
        # plt.gca().invert_xaxis()
        # plt.pause(0.05)
        # plt.show()

        cv2.waitKey(2)



def do_reperject():
    path = 'frame1.jpg'
    img = cv2.imread(path)
    H, W = img.shape[0], img.shape[1]

    F = 525
    K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
    Kinv = np.linalg.inv(K)

    lane_line = []
    for i in range(1,50):
        lane_line.append([1.8, 2.1, i])
    lane_line = np.array(lane_line)



    pix_lane = []
    for i in range(49):
        x, y, z = lane_line[0], lane_line[1], lane_line[2]
        # pix = np.dot(K, np.array([x/z, y/z, 1]))
        pix = np.dot(K, lane_line[i])
        pix = pix[:2] / pix[2]
        pix = [int(x) for x in pix]
        print(pix)

        cv2.circle(img, tuple(pix), 2, (0,255,0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)

def do_3dtobev():
    videp_path = 'output.mp4'
    cap = cv2.VideoCapture(videp_path)

    H, W = 374, 1242
    F = 525
    K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
    Kinv = np.linalg.inv(K)

    lane_line = []
    for i in range(1, 50):
        lane_line.append([1.8, 2.1, i])
    lane_line = np.array(lane_line)

    bev_size = (1000, 800)
    bev = np.zeros(bev_size)

    while cap.isOpened():
        ret, img = cap.read()


        for i in range(49):
            pix = np.dot(K, lane_line[i])
            pix = pix[:2] / pix[2]
            pix = [int(x) for x in pix]
            print(pix)
            cv2.circle(img, tuple(pix), 2, (0, 255, 0), 2)

        bev = bev_trans(bev, lane_line, bev_size)

        cv2.imshow('img', img)
        cv2.imshow('bev', bev)
        cv2.waitKey(2)

def bev_trans(img, lane, bev_size):
    H, W = bev_size[0], bev_size[1]
    cx, cy = int(W/2), H

    for i in range(49):
        [x, y, z] = [e for e in lane[i]]
        pix_x = int(cx + x)
        pix_y = int(cy - z) - (i * 5)

        print('bev', pix_x, pix_y, x, z)
        cv2.circle(img, (pix_x, pix_y), 2, (255,255,255), 2)
    return img

if __name__ == '__main__':
    # do_bev()
    # ImgUt.read_img_xy('tst.jpg')
    # do_reperject()
    do_3dtobev()