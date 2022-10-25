from ast import Gt, arg
import sys
import os
import argparse
import time

import numpy as np

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

import torch.utils.model_zoo as model_zoo
import torchvision

from model.sixdrepnet  import SixDRepNet
import datasets 
import loss 

from torchinfo import summary
import sys

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='物体の姿勢を推定するネットワーク')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',help='学習回数',default=100, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='batchサイズ',default=64, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='学習率',default=0.0001, type=float)
    parser.add_argument(
        '--scheduler', default=False , type=bool)
    parser.add_argument(
        '--train_dir', dest='train_dir', help='Directory path for data.', type=str)
    parser.add_argument(
        '--val_dir', dest='val_dir', help='Directory path for data.', type=str)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    #コマンドライン引数の解析
    args = parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler

    #model
    model = SixDRepNet(gpu)
    model.cuda(gpu)
    #Data_Augmentation
    transformations = datasets.DataTransform()
    #dataset
    pose_dataset = datasets.Pose_Dataset(str(args.train_dir), str(args.val_dir) , transformations(), train=True)

    # summary(model=model, input_size=(batch_size, 3, 256, 256))
    # sys.exit()
    #data_loader    
    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        num_workers=1
        )


    crit = loss.GeodesicLoss().cuda(gpu) 
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    milestones = [10, 20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    print('Starting training.')
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0
        for i, (gt_mat , images) in enumerate(train_loader):
            iter += 1 
            images = images.cuda(gpu)
            # Forward pass
            pred_mat = model(images)

            # Calc loss
            loss = crit(gt_mat.cuda(), pred_mat)
            
            optimizer.zero_grad()#勾配を初期化
            loss.backward() #逆伝播
            optimizer.step()#パラメータ(重み)の更新
            loss_sum += loss.item() 
            print(f"now_epock_{epoch},batch_{i}")
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(pose_dataset)//batch_size}] Loss:{loss.item()} ')
            if b_scheduler :
                scheduler.step()

        if epoch%100 == 0 :
           torch.save(model.state_dict(), f"./weight/progress/progressstep{epoch}_mdoel.pth") 
           ## 学習途中の状態を保存する。
           #torch.save({"epoch": epoch,"model_state_dict": model.state_dict(),"optimizer_state_dict": optimizer.state_dict(),},"model.tar",)
    torch.save(model.state_dict(), "./weight/result/last_model.pth") 
