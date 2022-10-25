
import math
from matplotlib import pyplot as plt
import sys
import os
import argparse

import numpy as np
import cv2


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import SixDRepNet.utils
import matplotlib
from PIL import Image
import time
matplotlib.use('TkAgg')
import model.utils
from model.sixdrepnet  import SixDRepNet

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cam',
                        dest='cam_id', help='Camera device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


transformations = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    cam = args.cam_id
    snapshot_path = args.snapshot
    model = SixDRepNet(gpu)
    saved_state_dict = torch.load("./weight/result/last_model.pth")
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()  

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()

            #物体検出
            for box, landkmarks, score in faces:

                # Print the location of each face in this image
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transformations(img)

            img = torch.Tensor(img[None, :]).cuda(gpu)

            c = cv2.waitKey(1)
            if c == 27:
                break

            R_pred = model(img)
            euler = model.utils.compute_euler_angles_from_rotation_matrices(
                R_pred)*180/np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            #utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
            # model.utils.plot_pose_cube(frame,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
            #     x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)

            # cv2.imshow("pred_deg", frame)
            # cv2.waitKey(5)
