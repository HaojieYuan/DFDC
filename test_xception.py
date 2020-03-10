


import sys
sys.setrecursionlimit(15000) #设置最大递归层数
import os
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data
from dataset.Dfirstdataset import faceforensicsDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import torchvision.models as models
from network.models import TransferModel


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='databases/faceforensicspp', help='path to dataset')
parser.add_argument('--test_set', default ='test', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/binary_faceforensicspp', help='folder to output model checkpoints')
#parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=21, help='checkpoint ID')

opt = parser.parse_args()
print(opt)



if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    #dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    dataset_test = faceforensicsDataset(rootpath=opt.dataset, datapath=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))


    model = TransferModel(modelchoice='xception', num_out_classes=1)
    #model.set_trainable_up_to(False) #设置fc为线性
    if opt.gpu_id >= 0:
        model = nn.DataParallel(model) #设置并行计算
        model.cuda()


    model.load_state_dict(torch.load(os.path.join(opt.outf,'model_' + str(opt.id) + '.pt')))
    model.eval()

    # if opt.gpu_id >= 0:
    #     model.cuda()



    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    count = 0
    loss_test = 0

    for img_data, labels_data in tqdm(dataloader_test):

        labels_data[labels_data > 1] = 1
        img_label = labels_data.numpy().astype(np.float)

        if opt.gpu_id >= 0:
            img_data = img_data.cuda()
            labels_data = labels_data.cuda()

        input_v = Variable(img_data)

        classes = model(input_v)

        classes = classes.view(1,-1).squeeze(0)
        classes = torch.sigmoid(classes)

        output_dis = classes.data.cpu()
        # output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

        # for i in range(output_dis.shape[0]):
        #     if output_dis[i,1] >= output_dis[i,0]:
        #         output_pred[i] = 1.0
        #     else:
        #         output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_dis))
        
        # pred_prob = torch.softmax(output_dis, dim=1)
        # tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:,1].data.numpy()))

        count += 1

    logloss_test = metrics.log_loss(tol_label, tol_pred)
    # log_loss = metrics.log_loss(tol_label, tol_pred_prob)
    # pre_test = metrics.precision_score(tol_label, tol_pred, average=None)
    # recall_test = metrics.recall_score(tol_label, tol_pred, average=None)
    loss_test /= count

    # fpr, tpr, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # fnr = 1 - tpr
    # hter = (fpr + fnr)/2

    # print('log_loss:%.4f'%(log_loss))
    # print('precision:', pre_test)
    # print('recall:', recall_test)
    print('[Epoch %d] Test logloss: %.4f' % (opt.id, logloss_test))
    text_writer.write('%d,%.4f\n'% (opt.id, logloss_test))

    text_writer.flush()
    text_writer.close()


        

