"""
加载每一帧的第一个人脸
"""

import os 
import glob 
import torch
import pandas as pd
#from skimage import io
from PIL import Image
from PIL import ImageFile
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torchvision import transforms, utils

from albumentations import GaussianBlur, JpegCompression, GaussNoise, HorizontalFlip, Compose

ImageFile.LOAD_TRUNCATED_IMAGES = True

def aug():
    return Compose([
            HorizontalFlip(p=0.5),
            GaussianBlur(blur_limit=7, p=0.5),
            JpegCompression(quality_lower=39, quality_upper=40, p=0.5),
            ])

class faceforensicsDataset(Dataset):

    def __init__(self, rootpath, datapath, transform=None):
        #rootpath="/mnt/lvdisk1/miaodata/" datapath="manipulated_sequences/Deepfakes/c23/faces/000_003/"
        imgsfolderPath = open(datapath,'r')
        imgs = []
        for line in imgsfolderPath:
            line = line.rstrip()
            words = line.split()
            filelist = glob.glob(os.path.join(rootpath, words[0], '*_0.png'))
            #print(len(filelist))
            for imagepath in filelist:
                imgs.append((imagepath, int(words[1])))

        self.imgs = imgs
        #self.rootpath
        self.transform = transform


    def __len__(self):
        #print(len(self.imgs))
        return(len(self.imgs))


    def __getitem__(self,idx):
        img_name, label = self.imgs[idx]
        #image = Image.open(img_name).convert('RGB')
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmentation = aug()
            image = augmentation(image=image)['image']
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label



if __name__ == "__main__":
    face_dataset = faceforensicsDataset(rootpath='/mnt/lvdisk1/miaodata/DFDC/firstfaces_align/', datapath='/mnt/lvdisk1/miaodata/DFDC/firstfaces_align/train_first.txt')
    print(len(face_dataset))
    sample = face_dataset[1]
    print(sample)
