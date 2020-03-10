"""
删除损坏的视频帧
"""
import os 
import imghdr 
#from progressbar import ProgressBar 
from tqdm import tqdm
import argparse


p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
p.add_argument('--data_path', type=str)
args = p.parse_args()



#path ='H:\\datasets\\dogsandcats\\' 

original_images =[] 

#os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
for root, dirs, filenames in os.walk(args.data_path):     
    for filename in filenames:         
        original_images.append(os.path.join(root, filename)) 



original_images = sorted(original_images) 
print('num:',len(original_images)) 

#f = open('check_error.txt','w+') 

error_images =[] 
#progress = ProgressBar() 


for filename in tqdm(original_images):     
    check = imghdr.what(filename)  

    if check == None:         
        # f.write(filename)         
        # f.write('\n')         
        error_images.append(filename) 
        print(filename)
        os.remove(filename)
                


print('error_images_num:',len(error_images)) 

# f.seek(0) 
# for s in f:    
#     print(s) 
# f.close()
