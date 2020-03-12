import os
import random

list_path = './fake_match_list.txt'
error_path = './error_list.txt'
out_path = './real_match_list.txt'
out_path2 = './balance_no_match.txt'
f = open(list_path)
error_f = open(error_path)
t1 = open(out_path, 'w')
t2 = open(out_path2, 'w')

real_list = []
error_list = []

for line in error_f:
    error_file = line.strip()
    error_list.append(error_file)


for line in f:
    fake, _, real = line.strip().split()
    if fake in error_list or real in error_list:
        continue
    tail = '/000_0.png'

    # make sure every fake could find its correspnding frame
    timer = 0
    NOT_FOUND = False
    while not os.path.exists(os.path.join('/mnt/lvdisk1/miaodata/DFDC/allfaces',fake+tail)) \
        or not os.path.exists(os.path.join('/mnt/lvdisk1/miaodata/DFDC/allfaces',real+tail)) \
        or real+tail in real_list:
        rand_num = random.randint(0, 299)
        tail = '/%03d_0.png'%(rand_num)
        timer = timer + 1
        if timer >= 100:
            NOT_FOUND = True
            break
    
    if NOT_FOUND:
        print(fake, real)
        continue
    
    real = real+tail
    fake = fake+tail
    real_list.append(real)

    t1.write(fake+' '+real+'\n')
    t2.write(fake+' '+'1'+'\n')
    t2.write(real+' '+'0'+'\n')

f.close()
error_f.close()
t1.close()
t2.close()