import cv2
import numpy as np
from atutils.imgu import read_img_xy



def read_xy():
    path = 'frame2.jpg'
    img = cv2.imread(path)
    read_img_xy(img)




if __name__ == '__main__':
    read_xy()