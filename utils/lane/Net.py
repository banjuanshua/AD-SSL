import cv2
import os
import torch as t
import torch.nn as nn

import numpy as np




class JLNet(nn.Module):
    def __init__(self):
        # todo auto calib
        super(JLNet, self).__init__()
        F = 525
        W = 1024
        H = 308
        K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
        self.K = t.Tensor(K)


    def forward(self, x):
        pass



    def rep_criterion(self, img_pts, pts3d):
        pts2d = []
        for i in range(50):
            pix = np.dot(self.K, pts3d[i])
            pix = pix[:2] / pix[2]
            pts2d.append(pix)


        pts2d = np.array(pts2d)
        pts2d = t.Tensor(pts2d)

        loss_x = (pts2d[:, 0]-img_pts[:, 0]) ** 2
        loss_y = (pts2d[:, 1]-img_pts[:, 1]) ** 2
        loss = sum(loss_x) + sum(loss_y)
        loss = t.clamp(loss, 0, 50)

        return loss

def rep_criterion_test():
    jlnet = JLNet()
    path = 'frame1.jpg'
    img = cv2.imread(path)
    print(img.shape)

    img_pts = np.array([[100, 100] for _ in range(50)])
    img_pts = t.Tensor(img_pts)

    pts3d = []
    for i in range(1, 51):
        pts3d.append([1.8, 2.1, i])
    pts3d = np.array(pts3d)
    pts3d = t.Tensor(pts3d)


    rep_loss = jlnet.rep_criterion(img_pts, pts3d)

    print(rep_loss)


def JLNet_test():
    jlnet = JLNet()
    path = 'frame1.jpg'
    img = cv2.imread(path)
    print(img.shape)

    inputs = t.Tensor([img])
    outputs = jlnet(inputs)

    print(outputs)




if __name__ == '__main__':
    rep_criterion_test()
    # JLNet_test()