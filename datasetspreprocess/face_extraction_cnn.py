"""
面部特征提取
"""
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image
from tqdm import tqdm_notebook
import cv2
import dlib
import numpy as np

class FaceExtraction:
    
    def __init__(self):
        self.face_detector = dlib.cnn_face_detection_model_v1('/home/mct/faceforensics++/miaofaceforensics-train/datasetspreprocess/mmod_human_face_detector.dat') 
        #CNN人脸检测器
        
    def extract_face(self,image_path,image_shape=None,to_gray=True,print_not_found=False):
        image  = cv2.imread(image_path) #读取单个图片
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if to_gray:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(image, 1) #1表示采样次数，获取人脸,参数
        if len(faces):
            for i, d in enumerate(faces):
                if i == 0:
                    face = d.rect
                    
            # face = faces[0]
            height, width = image.shape[:2] #图片通道数
            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = FaceExtraction.get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]
            if image_shape is not None:
                cropped_face = FaceExtraction.preprocess(cropped_face,image_shape)
                #cropped_face = FaceExtraction.preprocess(cropped_face)
            return cropped_face
        else:
            if print_not_found:
                print('Face not found in image {}'.format(image_path))
            return None
        
    @staticmethod    
    def preprocess(image, image_shape):


        image = cv2.resize(image,tuple(image_shape),interpolation=cv2.INTER_LINEAR)#双线性插值
        image = np.float32(image)
        image /= 255.0 
        return image
    
        
    @staticmethod
    def get_boundingbox(face, width, height, scale=1.3, minsize=None):
        """
        Expects a dlib face to generate a quadratic bounding box.
        :param face: dlib face class
        :param width: frame width
        :param height: frame height
        :param scale: bounding box size multiplier to get a bigger face region
        :param minsize: set minimum bounding box size
        :return: x, y, bounding_box_size in opencv form
        """
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        if minsize:
            if size_bb < minsize:
                size_bb = minsize
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Check for out of bounds, x-y top left corner
        x1 = max(int(center_x - size_bb // 2), 0)
        y1 = max(int(center_y - size_bb // 2), 0)
        # Check for too big bb size for given x, y
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        return x1, y1, size_bb



