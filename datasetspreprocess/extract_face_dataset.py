"""
提取人脸区域，并保存对应文件夹
"""

import os
from glob import glob
import cv2
import dlib
#from face_extraction import FaceExtraction
from face_extraction_cnn import FaceExtraction
import argparse
from tqdm import tqdm
from os.path import join

DATASET_PATHS = {
    'youtube': 'original_sequences/youtube',
    'actors': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection'
}
COMPRESSION = ['raw', 'c23', 'c40']




def detect_store_face(image_path):
    store_path = image_path.replace('/images/','/faces/')
    #if not os.path.exists(store_path):
    store_folder = '/'.join(store_path.split('/')[:-1])
    os.makedirs(store_folder,exist_ok=True)
    #os.makedirs(store_folder)
    
    face = FE.extract_face(image_path,to_gray=False,print_not_found=True)
    if face is not None:
        cv2.imwrite(store_path,face)#第一个参数保存的文件名，第二个保存的图像


def extract_method_images(data_path, dataset, compression):
    """Extracts all images of a specified method and compression in the
    FaceForensics++ file structure"""
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
    #faces_path = join(data_path, DATASET_PATHS[dataset], compression, 'faces')
    #os.listdir返回指定路径下的文件和文件夹列表
    for images in tqdm(os.listdir(images_path)):
        #face_folder = image.split('.')[0]
        #extract_frames(join(images_path, image),
                       #join(faces_path, face_folder))
        for image in os.listdir(join(images_path,images)):
            detect_store_face(join(images_path,images,image))




if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str) #数据集所在根目录
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all') #用--dataset，-d为简写
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='raw') #用-c
    args = p.parse_args()

    FE = FaceExtraction()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_images(**vars(args))
    else:
        extract_method_images(**vars(args))


