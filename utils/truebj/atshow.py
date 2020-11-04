import os
import parseTrackletXML as xmlParser
import numpy as np
import pykitti
from PIL import Image
import cv2

from cfg import *



def plot_3dbox():
    tracklets = xmlParser.example(kittiDir, drive)
    data = pykitti.raw('/Volumes/hardware/KITTI/raw_data', '0027', '2011_09_26', frames=range(0, 50, 5))

    for iTracklet, tracklet in enumerate(tracklets):
        print('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))

        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet\
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]])

        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber \
                in tracklet:
            if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
                continue

            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are 0 in all xml files I checked
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T


            x, y, z = translation
            yawVisual = (yaw - np.arctan2(y, x)) % twoPi



            # 读图片
            img_id = ''.join(['0' for _ in range(10-len(str(absoluteFrameNumber)))]) + str(absoluteFrameNumber)
            img_path = '/Volumes/hardware/KITTI/raw_data/2011_09_26/2011_09_26_dri' \
                       've_0027_sync/show/data/' + img_id + '.png'
            img = cv2.imread(img_path)

            # 转坐标系
            x_arr = cornerPosInVelo[0]
            y_arr = cornerPosInVelo[1]
            z_arr = cornerPosInVelo[2]
            draw_points = []
            for i in range(len(x_arr)):
                velo_point = [x_arr[i], y_arr[i], z_arr[i], 1]
                # velo_point = [0,0,0,1]
                cam0_point = data.calib.T_cam0_velo.dot(velo_point)
                cam0_point = data.calib.R_rect_20.dot(cam0_point)

                x, y, z = cam0_point[0], cam0_point[1], cam0_point[2]
                cam_point = data.calib.K_cam2.dot([x / z, y / z, 1])

                draw_points.append((int(cam_point[0]), int(cam_point[1])))

                # print(i, velo_point, cam_point)
                # cv2.circle(img, (int(cam_point[0]), int(cam_point[1])), point_size, point_color, thickness)

            # 画框
            line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                          [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])
            for order in line_order:
                cv2.line(img, draw_points[order[0]], draw_points[order[1]], (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(img_path, img)

            print(img.shape)
            break

def img2video():
    floder_path = '/Volumes/hardware/KITTI/raw_data/2011_09_26/2011_09_26_dri' \
                       've_0027_sync/show/data/'
    files_raw = os.listdir(floder_path)
    files_raw.sort()
    files = [file for file in files_raw if '_' not in file]

    fps = 10
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    vout = cv2.VideoWriter()
    vout.open('output.mp4', fourcc, fps, (1242, 375))

    for file in files:
        img_path = floder_path + file
        img = cv2.imread(img_path)
        vout.write(img)

        print(file)

    vout.release()
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    # plot_3dbox()
    img2video()