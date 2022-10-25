import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter
import glob
import model.utils
import numpy as np
    
class Pose_Dataset(Dataset):
    def __init__(self, img_dir, label_dir , transform, train=False):
        self.img_list=self.make_datapath_list(img_dir)
        self.label_list=self.make_datapath_list(label_dir)
        self.transform = transform
        self.phase=train

    def __getitem__(self,index): #index指定したときに実行される
        img_path = self.img_list[index]
        label_path = self.label_list[index]
        if self.transform is not None and self.phase : 
            #画像の読み込み
            img_data = Image.open(img_path)
            img_data = self.transform(img_data)
            #ラベルデータの読み込み
            label_data = np.loadtxt(label_path,dtype="float16")
            label_data = model.utils.get_R(label_data[0],label_data[1],label_data[2])
            #numpy→tensor変換
            label_data = torch.from_numpy(label_data).float()
        
        return label_data , img_data 

    
    def __len__(self):
        return len(self.img_list)

    def make_datapath_list(self, data_dir):
        data_list = glob.glob(data_dir+'/*')
        return data_list
        

class DataTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    def __call__(self):
        return self.data_transform
