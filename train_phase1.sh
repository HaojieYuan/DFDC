CUDA_VISIBLE_DEVICES=4,5,6,7 python train_phase1_binary.py --dataset /mnt/lvdisk1/miaodata/DFDC/firstfaces_align/ \
--train_set /home/zhangjiabin/faceforensics/haojie/DFDC/fake_match_list.txt \
--val_set /mnt/lvdisk1/miaodata/DFDC/firstfaces_align/valid_4049.txt \
--workers 4 --batchSize 128 \
--outf /mnt/lvdisk1/miaodata/DFDCcode/ckpt_databalancing/phase1/ --manualSeed 8664