"""
设置训练、验证、测试数据集，720、140、140

"""


from glob import glob
import os
import numpy as np
#import matplotlib.pyplot as plt
import shutil
# from torchvision import transforms
# from torchvision import models
# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# from torch.optim import lr_scheduler
# from torch import optim
# from torchvision.datasets import ImageFolder
# from torchvision.utils import make_grid
# import time

path1 = '/data/'

os.mkdir(os.path.join(path1,'train'))
os.mkdir(os.path.join(path1,'valid'))
os.mkdir(os.path.join(path1,'test'))

for t in ['train','valid','test']:
    for folder in ['manipulated_sequences/','original_sequences/']:
        os.mkdir(os.path.join(path1,t,folder))




#"/home/mct/faceforensics++/original_sequences/youtube/c23/images/999/0333.png"






path2 = '/home/mct/faceforensics++/'

#vid_dir = os.listdir(path2)

#" * "对于文件夹表示文件所在的目录的文件夹，对于文件是匹配0个或者多个字符（文件名）
files = glob(os.path.join(path2,'*/*/*/*/*/*.png'))

print(f'Total no of images{len(files)}')

no_of_images = len(files)
a = int(no_of_images * 0.72)
b = int(no_of_images * 0.86)

shuffle = np.random.permutation(no_of_images)

for i in shuffle[:a]:
    folder = files[i].split('/')[4]
    images = '_'.join((files[i].split('/')[5],files[i].split('/')[6],files[i].split('/')[-1]))
    shutil.copyfile(files[i],os.path.join(path1,'train',folder,images))

for i in shuffle[a:b]:
    folder = files[i].split('/')[4]
    images = '_'.join((files[i].split('/')[5],files[i].split('/')[6],files[i].split('/')[-1]))
    shutil.copyfile(files[i],os.path.join(path1,'valid',folder,images))

for i in shuffle[b:]:
    folder = files[i].split('/')[4]
    images = '_'.join((files[i].split('/')[5],files[i].split('/')[6],files[i].split('/')[-1]))
    shutil.copyfile(files[i],os.path.join(path1,'test',folder,images))


