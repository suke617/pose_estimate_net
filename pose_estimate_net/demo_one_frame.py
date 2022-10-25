import math
from matplotlib import pyplot as plt
import numpy as np
import cv2


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import time
import model.utils as utils
from model.sixdrepnet  import SixDRepNet    


transformations = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


if __name__ == '__main__':
    gpu = 0
    model = SixDRepNet(gpu)
    saved_state_dict = torch.load("./weight/result/last_model.pth")
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()  
    img_path = "./dataset/img_data/pic0.png"
    img = Image.open(img_path) 
    img = transformations(img)
    with torch.no_grad():
        img = img.cuda(gpu)
        start = time.time()
        R_pred = model(img.unsqueeze(0))
        end = time.time()
        print('execution_time_ %2f s' % ((end - start)))
        euler = utils.compute_euler_angles_from_rotation_matrices(
        R_pred)*180/np.pi
        p_pred_deg = euler[:, 0].cpu()
        y_pred_deg = euler[:, 1].cpu()
        r_pred_deg = euler[:, 2].cpu()
        print(p_pred_deg,y_pred_deg,r_pred_deg)
