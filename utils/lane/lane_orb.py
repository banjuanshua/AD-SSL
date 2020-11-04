import cv2
from atutils.imgu import *



if __name__ == '__main__':
    videp_path = '../output.mp4'
    cap = cv2.VideoCapture(videp_path)

    dect = [[450, 371],
            [775, 371],
            [450, 184],
            [775, 184]]

    while (cap.isOpened()):
        ret, img = cap.read()
        # img = img.crop((450, 184, 775, 371))
        img = img[184:371, 450:775]
        img, kps, des = extract_orb(img)
        cv2.imshow('img',img)
        cv2.waitKey(0)