"""

"""
import os
import numpy as np
import argparse

def text_save(filepath, path, label):
    folder = os.listdir(os.path.join(filepath, path))

    no_of_images = len(folder)
    a = int(no_of_images * 0.72)
    b = int(no_of_images * 0.86)

    shuffle = np.random.permutation(no_of_images)

    trainfile = open(os.path.join(filepath,'train.txt'),'a')
    for i in shuffle[:a]:
        s_train = path + str(folder[i]) + ' ' + label + '\n' 
        trainfile.write(s_train)
    trainfile.close()


    validfile = open(os.path.join(filepath,'valid.txt'),'a')
    for i in shuffle[a:b]:
        s_valid = path + str(folder[i]) + ' ' + label + '\n' 
        validfile.write(s_valid)
    validfile.close()


    testfile = open(os.path.join(filepath,'test.txt'),'a')
    for i in shuffle[b:]:
        s_test = path + str(folder[i]) + ' ' + label + '\n' 
        testfile.write(s_test)
    testfile.close()


#"/mnt/lvdisk1/miaodata/manipulated_sequences/Face2Face/c23/images/"
#"/mnt/lvdisk1/miaodata/original_sequences/youtube/c23/images/"

if __name__ == "__main__":

    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    p.add_argument('--data_path', type=str)
    args = p.parse_args()
    
    filepath = args.data_path
    
    path = 'original_sequences/youtube/c23/faces/'
    label = '0'
    text_save(filepath, path, label)

    path = 'manipulated_sequences/Face2Face/c23/faces/'
    label = '1'
    text_save(filepath, path, label)

    path = 'manipulated_sequences/Deepfakes/c23/faces/'
    label = '1'
    text_save(filepath, path, label)

    path = 'manipulated_sequences/FaceSwap/c23/faces/'
    label = '1'
    text_save(filepath, path, label)

    path = 'manipulated_sequences/NeuralTextures/c23/faces/'
    label = '1'
    text_save(filepath, path, label)




