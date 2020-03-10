import os 
# import sys
# sys.path.append("/mnt/lvdisk1/miaodata/DFDCcode/facenet/models/")
import glob 
import torch
import pandas as pd
import cv2 
from PIL import Image
from PIL import ImageFile
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torchvision import transforms, utils
from facenet import MTCNN

ImageFile.LOAD_TRUNCATED_IMAGES = True

class videoDataset(Dataset):

    def __init__(self, rootpath, datapath, transform=None):
        
        videosfolderPath = open(datapath, 'r')
        videos = []
        for line in videosfolderPath:
            line = line.rstrip()
            words = line.split()
            videos.append((os.path.join(rootpath, words[0]), words[1]))

        self.videos = videos
        self.transform = transform

    def __len__(self): 
        return(len(self.videos))

    def __getitem__(self, idx):
        video, label = self.videos[idx]

        sample = extract_faces(video)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

def extract_faces(videopath):

        mtcnn = MTCNN(device='cuda', image_size=299, margin=0.3/1.3*299).eval()
        reader = cv2.VideoCapture(videopath)
        frame_list = []
        frame_num = 0
        while reader.isOpened():
            success, frame = reader.read()
            if not success:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(image)
            frame_num += 1
        reader.release()
        # print(len(frame_list[:10]))
        faces = []
        with torch.no_grad():
            face = mtcnn(frame_list[:2])
            faces.extend(face)
            faces = [f for f in faces if f is not None]

        return faces[0]

if __name__ == "__main__":
    face_dataset = videoDataset(rootpath='/mnt/lvdisk1/miaodata/DFDC/videos/', datapath='/mnt/lvdisk1/miaodata/DFDC/videos/train_video.txt')
    print(len(face_dataset))
    sample = face_dataset[1]
    print(sample)