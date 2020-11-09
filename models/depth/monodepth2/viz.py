# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image

import torch
import numpy
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

from monotools import *
import matplotlib.pyplot as plt



def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--video_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()



def image2bev(img_raw, depth, K):
    img_bev = np.zeros((192, 150, 3))
    img_shape = depth.shape

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            # todo i在前还是j在前
            p2d = np.array([j, i, 1], dtype=np.float32)
            print(i, j, depth[i][j], p2d)
            p2d *= depth[i][j]
            invert_K = np.linalg.pinv(K)
            p3d = np.dot(invert_K, p2d)

            x, y, z = p3d[0], p3d[1], p3d[2]
            # print(p3d)
            if 0<=x<192 and 0<z<200:
                x, z = int(x), int(z)
                # print(x, y, z, i, j)
                # print(img_bev[x][x], img_raw[i][j])

                img_bev[x,z,:] = img_raw[i,j,:]

    return img_bev


    # points3d = []

    # u_arr = []
    # v_arr = []
    # z_arr = []

    # for x, y in zip(x_arr, y_arr):
    #     z = depth[y][x]
    #     p2d = np.array([x, y, 1], np.float32)
    #     p2d *= z

    #     u_arr.append(p2d[0])
    #     v_arr.append(p2d[1])
    #     z_arr.append(z)

    #     invert_K = np.linalg.pinv(K)
    #     p3d = np.dot(invert_K, p2d)
    #     if p3d[2] < 100:
    #         p3d = Point3d(p3d[0], p3d[1], p3d[2])
    #         points3d.append(p3d)



def video_viz(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()


    calib = Calibration('000000.txt')
    cap = cv2.VideoCapture(args.video_path)

    fig = plt.figure()

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        # for idx, image_path in enumerate(paths):
        while(cap.isOpened()):
            ret, frame = cap.read()

            # convert cv2 to PIL
            input_image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))  

            # Load image and preprocess
            # input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            img_raw = cv2.cvtColor(numpy.asarray(input_image),cv2.COLOR_RGB2BGR) 

            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)


            # Saving colormapped depth image
            # disp_resized_np = disp_resized.squeeze().cpu().numpy()
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            disp_resized_np = disp.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            
            img_depth = cv2.cvtColor(numpy.asarray(im),cv2.COLOR_RGB2BGR)
            depth *= 100
            depth = depth.squeeze().numpy()
            # img_bev = image2bev(img_raw, depth, K)

            # name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            # im.save(name_dest_im)

            points = project_depth_to_points(calib, depth, max_high=3)

            
            ax = fig.add_subplot(111, projection='3d')

            
            x = []
            y = []
            z = []
            for i in range(len(points)):
                x.append(points[i][0])
                y.append(points[i][1])
                z.append(points[i][2])
            
            
            

            cv2.imshow('raw', img_raw)
            cv2.imshow('depth', img_depth)
            # cv2.imshow('bev', img_bev)


            ax.scatter(x, y, z, c='k', marker='.', s=0.1)
            plt.pause(1)
            plt.clf()

            cv2.waitKey(1)

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    video_viz(args)
