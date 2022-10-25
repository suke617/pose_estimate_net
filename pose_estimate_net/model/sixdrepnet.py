import torch
from torch import nn
import torch.nn.functional as F
import math
#import utils
import cv2

import sys
import os
import argparse
import time
from PIL import Image

import numpy as np

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

import torch.utils.model_zoo as model_zoo
import torchvision
from model.utils import compute_rotation_matrix_from_ortho6d


"""
@autor 吉田圭佑
original_model : https://github.com/thohemp/6DRepNet
このネットワークのパラメータ変更点
IN_CHANNELS,OUT_CHANNELS,STRIDES(conv層のinputとoutputは合わせる事)


"""

class RepVGG(nn.Module):
    def __init__(self,in_channels, out_channels, stride, skip_connect=False):
        super(RepVGG, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.stride=stride
        self.nonlinearity = nn.ReLU()
        self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) 
        padding_11 = 1 - 3  // 2
        self.rbr_dense = self.conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=self.stride, padding=1, )
        self.rbr_1x1 = self.conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self.stride, padding=padding_11 ) 
        self.skip_connect = skip_connect
        

        self.se = nn.Identity() #中間層の特徴ベクトルを取り出すために恒等関数にする

    def forward(self, x):
        # #初回はスキップ接続なし
        if self.skip_connect :
            #スキップ結合 #3×3conv,1×1conv,batch_normalizationの出力を結合
            return self.nonlinearity(self.se(self.rbr_dense(x) + self.rbr_1x1(x))) #+ self.rbr_identity(x)
        
        else:
            #スキップ結合 #3×3conv,1×1conv,batch_normalizationの出力を結合
            return self.nonlinearity(self.se(self.rbr_dense(x) + self.rbr_1x1(x)))
            


    def conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        #畳み込み層＋batch正規化
        conv=nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        bn = nn.BatchNorm2d(num_features=out_channels)
        result = nn.Sequential(conv,bn)
        return result



class SixDRepNet(nn.Module):
    IN_CHANNELS=[3,16,32,64,128] #,256,512 #RepVGGの入出力
    OUT_CHANNELS=[16,32,64,128,256] #512,2048
    STRIDES = [2, 5, 7, 17, 2] 
    def __init__(self , gpu_id=0):
        super(SixDRepNet, self).__init__()
        self.gpu_id = gpu_id

        self.layer0 = RepVGG(in_channels=self.IN_CHANNELS[0] , out_channels=self.OUT_CHANNELS[0] ,stride=self.STRIDES[0] ,skip_connect=False) 
        self.layer1 = RepVGG(in_channels=self.IN_CHANNELS[1] , out_channels=self.OUT_CHANNELS[1] ,stride=self.STRIDES[1] ,skip_connect=True)
        self.layer2 = RepVGG(in_channels=self.IN_CHANNELS[2] , out_channels=self.OUT_CHANNELS[2] ,stride=self.STRIDES[2] ,skip_connect=True)
        self.layer3 = RepVGG(in_channels=self.IN_CHANNELS[3] , out_channels=self.OUT_CHANNELS[3] ,stride=self.STRIDES[3] ,skip_connect=True)
        self.layer4 = RepVGG(in_channels=self.IN_CHANNELS[4] , out_channels=self.OUT_CHANNELS[4] ,stride=self.STRIDES[4] ,skip_connect=True)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        #最後の層の出力に合わせた全結合層を設定
        fea_dim = last_channel
        self.linear_reg = nn.Linear(fea_dim, 6)
 
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x= self.gap(x)
        x = torch.flatten(x,1) #全結合層に入力するために平坦化  #すべてのbatch数を0にする
        x = self.linear_reg (x) #全結合層に入力,オイラー角の出力
        if self.gpu_id ==-1:
            return compute_rotation_matrix_from_ortho6d(x, False, self.gpu_id)
        else:
            return compute_rotation_matrix_from_ortho6d(x, True, self.gpu_id)
        return x


# transformations = transforms.Compose([transforms.ToTensor(),
#                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# # #デバック用
# model=SixDRepNet(-1)
# #model.cuda()
# model.eval()
# # img=cv2.imread("June-02-2022-10-58-16-color.png")
# # img = img.convert('RGB')
# # img = transformations(img)
# img = Image.open("June-02-2022-10-58-16-color.png").convert('RGB')
# img = transformations(img)
# #img = torch.Tensor(img).cuda()
# # print(img.dim())
# # print(img.size())
# pred_mat = model(img.unsqueeze(0))
# #print(model)
# print(pred_mat)
# #print(pred_mat.size())

# euler = utils.compute_euler_angles_from_rotation_matrices(
#                     pred_mat)*180/np.pi
# print(euler)

