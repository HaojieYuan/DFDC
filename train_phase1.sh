CUDA_VISIBLE_DEVICES=2,3 python train_phase1_binary.py \
--dataset_train /mnt/lvdisk1/miaodata/DFDC/allfaces \
--dataset_test /mnt/lvdisk1/miaodata/DFDC/firstfaces_align/  \
--train_set /home/zhangjiabin/faceforensics/haojie/DFDC/get_list/real_match_list.txt \
--val_set /mnt/lvdisk1/miaodata/DFDC/firstfaces_align/valid_4049.txt \
--workers 4 --batchSize 64 --niter 5 \
--outf /mnt/lvdisk1/miaodata/DFDCcode/ckpt_databalancing_3_12_matching/phase1/ --manualSeed 8664