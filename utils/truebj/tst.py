import parseTrackletXML as xmlParser
import numpy as np
import pykitti
from PIL import Image
import cv2




def xml_test():
    TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
    TRUNC_IN_IMAGE = 0
    TRUNC_TRUNCATED = 1
    TRUNC_OUT_IMAGE = 2
    TRUNC_BEHIND_IMAGE = 3
    truncFromText = {'99': TRUNC_UNSET, '0': TRUNC_IN_IMAGE, '1': TRUNC_TRUNCATED,
                     '2': TRUNC_OUT_IMAGE, '3': TRUNC_BEHIND_IMAGE}
    twoPi = 2. * np.pi

    kittiDir = '/Volumes/hardware/KITTI/raw_data/2011_09_26'
    drive = '2011_09_26_drive_0027_sync'
    tracklets = xmlParser.example(kittiDir, drive)

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
            # print(absoluteFrameNumber)

            # determine if object is in the image; otherwise continue
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


            # calc yaw as seen from the camera (i.e. 0 degree = facing away from cam), as opposed to
            #   car-centered yaw (i.e. 0 degree = same orientation as car).
            #   makes quite a difference for objects in periphery!
            # Result is in [0, 2pi]
            x, y, z = translation
            yawVisual = (yaw - np.arctan2(y, x)) % twoPi


            if absoluteFrameNumber == 0:
                # print(absoluteFrameNumber, cornerPosInVelo)
                return cornerPosInVelo, translation


def pyk_test(box, translation):
    img_path = '/Volumes/hardware/KITTI/raw_data/2011_09_26/2011_09_26_drive_0027_sync/image_02/data/0000000000.png'
    img = cv2.imread(img_path)

    point_size = 2
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8


    basepath = '/Volumes/hardware/KITTI/raw_data'
    drive = '2011_09_26'
    date = '0027'
    data = pykitti.raw(basepath, date, drive, frames=range(0, 50, 5))

    # point_velo = [0,0,0,1]
    # point_cam2 = data.calib.T_cam2_velo.dot(point_velo)
    # print(point_cam2)

    x_arr = box[0]
    y_arr = box[1]
    z_arr = box[2]
    draw_points = []
    for i in range(len(x_arr)):
        velo_point = [x_arr[i], y_arr[i], z_arr[i], 1]
        # velo_point = [0,0,0,1]
        cam0_point = data.calib.T_cam0_velo.dot(velo_point)
        cam0_point = data.calib.R_rect_20.dot(cam0_point)

        x, y, z = cam0_point[0], cam0_point[1], cam0_point[2]
        cam_point = data.calib.K_cam2.dot([x/z, y/z, 1])

        draw_points.append((int(cam_point[0]), int(cam_point[1])))

        # print(i, velo_point, cam_point)
        # cv2.circle(img, (int(cam_point[0]), int(cam_point[1])), point_size, point_color, thickness)

    line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                  [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])

    for order in line_order:
        cv2.line(img, draw_points[order[0]], draw_points[order[1]], (0, 255, 0), 1, cv2.LINE_AA)


    cv2.imwrite('a.png', img)

if __name__ == '__main__':
    box, translation = xml_test()
    pyk_test(box, translation)