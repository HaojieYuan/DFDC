"""
Copyright (c) 2018, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for training XceptionNet
"""

import os
import random
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.optim import Adam
import torch.utils.data
import torchvision.datasets as dset
from datasetspreprocess.faceforensics_dataset import faceforensicsDataset
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import argparse
from sklearn import metrics
import gc
from network.models import TransferModel


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='databasess/FFPP', help='path to root dataset')
parser.add_argument('--train_set', default ='train', help='train set')
parser.add_argument('--val_set', default ='val', help='validation set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=299, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--eps', type=float, default=1e-08, help='epsilon')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='checkpoints/binary_phase2', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if __name__ == "__main__":

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.gpu_id >= 0:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    if opt.resume > 0:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'w')

    model = TransferModel(modelchoice='xception', num_out_classes=2)
    network_loss =  nn.CrossEntropyLoss()

    if opt.gpu_id >= 0:
        model = nn.DataParallel(model) #设置并行计算
        model.cuda()
        network_loss.cuda()
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), eps=opt.eps)

    if opt.resume > 0:
        model.load_state_dict(torch.load(os.path.join(opt.outf,'model_' + str(opt.resume) + '.pt')))
        optimizer.load_state_dict(torch.load(os.path.join(opt.outf,'optim_' + str(opt.resume) + '.pt')))

        if opt.gpu_id >= 0:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
    else:
        model.module.load_state_dict(torch.load(os.path.join(opt.outf,'model_0.pt'))) #
        model.module.set_trainable_up_to(True) #

    model.train(mode=True)

    # if opt.gpu_id >= 0:
    #     model.cuda()
    #     network_loss.cuda()

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    
    #dataset_train = dset.ImageFolder(root=os.path.join(opt.dataset, opt.train_set), transform=transform_fwd)
    dataset_train = faceforensicsDataset(rootpath=opt.dataset, datapath=os.path.join(opt.dataset, opt.train_set), transform=transform_fwd)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    #dataset_val = dset.ImageFolder(root=os.path.join(opt.dataset, opt.val_set), transform=transform_fwd)
    dataset_val = faceforensicsDataset(rootpath=opt.dataset, datapath=os.path.join(opt.dataset, opt.val_set), transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))


    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_train = 0
        loss_test = 0

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(dataloader_train):

            labels_data[labels_data > 1] = 1
            img_label = labels_data.numpy().astype(np.float)

            optimizer.zero_grad()
            if opt.gpu_id >= 0:
                img_data = img_data.cuda()
                labels_data = labels_data.cuda()

            classes = model(img_data)

            loss_dis = network_loss(classes, labels_data.data)
            loss_dis_data = loss_dis.item()

            loss_dis.backward()
            optimizer.step()

            output_dis = classes.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            for i in range(output_dis.shape[0]):
                if output_dis[i,1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_train += loss_dis_data
            count += 1

            # if count % 10000 == 0:
            #     torch.save(model.state_dict(), os.path.join(opt.outf, 'model_%d.pt' % epoch))
            #     torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch))



        acc_train = metrics.accuracy_score(tol_label, tol_pred)
        loss_train /= count

        ########################################################################

        # do checkpointing & validation
        torch.save(model.state_dict(), os.path.join(opt.outf, 'model_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch))

        model.eval()

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        count = 0

        for img_data, labels_data in tqdm(dataloader_val):

            labels_data[labels_data > 1] = 1
            img_label = labels_data.numpy().astype(np.float)

            if opt.gpu_id >= 0:
                img_data = img_data.cuda()
                labels_data = labels_data.cuda()

            classes = model(img_data)

            loss_dis = network_loss(classes, labels_data.data)
            loss_dis_data = loss_dis.item()
            output_dis = classes.data.cpu().numpy()

            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            for i in range(output_dis.shape[0]):
                if output_dis[i,1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_test += loss_dis_data
            count += 1

        acc_test = metrics.accuracy_score(tol_label, tol_pred)
        loss_test /= count

        print('[Epoch %d] Train loss: %.4f   acc: %.2f | Test loss: %.4f  acc: %.2f'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.flush()

        model.train(mode=True)

    text_writer.close()