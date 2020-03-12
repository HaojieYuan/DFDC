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
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torchvision import transforms, utils

ImageFile.LOAD_TRUNCATED_IMAGES = True
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
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class faceforensicsDatasetBalanced(Dataset):

    def __init__(self, rootpath, datapath, transform=None):
        #rootpath="/mnt/lvdisk1/miaodata/" datapath="manipulated_sequences/Deepfakes/c23/faces/000_003/"
        imgsfolderPath = open(datapath,'r')
        imgs_fake = []
        imgs_real = []
        for line in imgsfolderPath:
            line = line.strip()
            words = line.split()
            imagepath_fake = os.path.join(rootpath, words[0])
            imagepath_real = os.path.join(rootpath, words[1])
            #print(len(filelist))
            imgs_fake.append((imagepath_fake, 1))
            imgs_real.append((imagepath_real, 0))


        self.imgs_fake = imgs_fake
        self.imgs_real = imgs_real
        assert len(imgs_fake) == len(imgs_real)
        self.len = len(imgs_fake)
        #self.rootpath
        self.transform = transform


    def __len__(self):
        #print(len(self.imgs))
        return(self.len)


    def __getitem__(self,idx):
        img_name_fake, label_fake = self.imgs_fake[idx]
        img_name_real, label_real = self.imgs_real[idx]
        image_fake = Image.open(img_name_fake).convert('RGB')
        image_real = Image.open(img_name_real).convert('RGB')

        if self.transform:
            image_fake = self.transform(image_fake)
            image_real = self.transform(image_real)

        return [image_fake, image_real], [label_fake, label_real]

def balancing_collate_fn(data):
    """
       Defines how to batch up balanced sampling result pairs. 
       The input data will be list of tuples ([img_fake, img_real], [label_fake, label_real]).
    """

    imgs = [torch.stack(element[0]) for element in data]
    labels = [element[1] for element in data]

    imgs = torch.stack(imgs)
    imgs = imgs.reshape(-1, *(imgs.shape[2:])) # [m, n, p, ...] -> [m*n, p, ...]

    labels = torch.Tensor(labels)
    labels = labels.reshape(-1)

    return imgs.float(), labels.float()

if __name__ == "__main__":
    face_dataset = faceforensicsDataset(rootpath='/mnt/lvdisk1/miaodata/DFDC/firstfaces_align/', datapath='/mnt/lvdisk1/miaodata/DFDC/firstfaces_align/train_first.txt')
    print(len(face_dataset))
    sample = face_dataset[1]
    print(sample)
