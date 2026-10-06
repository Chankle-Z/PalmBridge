import os
import argparse
import time
import torch
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader

from models.dataset import MyDataset
from models.ccnet import ccnet
from models.ccnetvq import ccnetvq
from models.compnet import compnet
from models.compnetvq import compnetvq
from models.sfnet import SFNet
from models.sfnetvq import SFNetVq
from models.resnet import ResNet
from models.resnetvq import ResNetVq
from models.resnetDHN import ResNetDHN, Bottleneck, BasicBlock
from models.resDHNvq import ResNetDHNvq
from models.co3net import co3net
from models.co3netvq import co3netvq

from utils import *
import sys


def open_test_replace(model):
    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))



    # train_set_file = './data/train_IITD.txt'
    # test_set_file = './data/test_IITD.txt'

    galleryset = MyDataset(txt=gallery_set_file, transforms=None, train=False)
    queryset = MyDataset(txt=query_set_file, transforms=None, train=False)

    batch_size = 128  # 128

    data_loader_gallery = DataLoader(dataset=galleryset, batch_size=batch_size, num_workers=2)
    data_loader_query = DataLoader(dataset=queryset, batch_size=batch_size, num_workers=2)


    net = model

    net.cuda()
    net.eval()

    # feature extraction:

    start_time = time.time()
    replace_time = 0
    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_gallery):

        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        codes, time1 = net.getFeatureCode(data)
        replace_time = replace_time + time1
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]

    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_query):

        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        codes, time2 = net.getFeatureCode(data)

        replace_time = replace_time + time2

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    if batch_id != 1:
        print('aaaa')

    print('completed feature extraction.')
    print('featDB_test.shape: ', featDB_test.shape)

    print('\nFeature Extraction Done!')
    end_time = time.time()
    print("feature extraction time cost:", end_time - start_time)
    print("replace time cost:", replace_time)

def open_test(model):
    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))



    # train_set_file = './data/train_IITD.txt'
    # test_set_file = './data/test_IITD.txt'

    galleryset = MyDataset(txt=gallery_set_file, transforms=None, train=False)
    queryset = MyDataset(txt=query_set_file, transforms=None, train=False)

    batch_size = 128  # 128

    data_loader_gallery = DataLoader(dataset=galleryset, batch_size=batch_size, num_workers=2)
    data_loader_query = DataLoader(dataset=queryset, batch_size=batch_size, num_workers=2)


    net = model

    net.cuda()
    net.eval()

    # feature extraction:

    start_time = time.time()
    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_gallery):

        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]

    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_query):

        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    if batch_id != 1:
        print('aaaa')

    print('completed feature extraction.')
    print('featDB_test.shape: ', featDB_test.shape)

    print('\nFeature Extraction Done!')
    end_time = time.time()
    print("feature extraction time cost:", end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CO3Net for Palmprint Recfognition"
    )

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch_num", type=int, default=3000)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--weight1", type=float, default=0.8)
    parser.add_argument("--weight2", type=float, default=0.2)
    parser.add_argument("--loss_ce", type=float, default=0.7)
    parser.add_argument("--loss_tl", type=float, default=0.3)
    parser.add_argument("--com_weight", type=float, default=0.8)
    parser.add_argument("--id_num", type=int, default=378,
                        help="IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500 IITD_Right 230 Tongji_LR 300")
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--redstep", type=int, default=500)
    parser.add_argument("--vit_floor_num", type=int, default=10)

    parser.add_argument("--test_interval", type=str, default=1000)
    parser.add_argument("--save_interval", type=str, default=500)  ## 200 for Multi-spec 500 for RED
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gallery_set_file", type=str, default='/data/zck/CompVq/train_IITD_open_test_train.txt')
    parser.add_argument("--query_set_file", type=str, default='/data/zck/CompVq/test_IITD_open_test_test.txt')

    ##Store Path
    parser.add_argument("--checkpoint", type=str, default='./results/checkpoint/')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    num_classes = args.id_num
    comp_weight = args.com_weight
    gallery_set_file = args.gallery_set_file
    query_set_file = args.query_set_file
    loss_ce = args.loss_ce
    loss_tl = args.loss_tl
    vit_floor_num = args.vit_floor_num

    print("\n========== Test Config ==========")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Gallery Set: {args.gallery_set_file}")
    print(f"Query Set:   {args.query_set_file}")
    print("=================================\n")

    # model = ResNetDHNvq(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, gray=True)
    # model = ccnet(num_classes=num_classes, weight=comp_weight)
    # model = compnet(num_classes=num_classes)
    # model = SFNet(num_classes=num_classes,vit_floor_num=vit_floor_num)
    # model = ResNet(block=BasicBlock, layers=[2,2,2,2], num_classes=num_classes, gray=True)
    model = co3net(num_classes=num_classes)
    checkpoint = torch.load(args.checkpoint, map_location='cuda')
    model.load_state_dict(checkpoint)
    print("Loaded checkpoint successfully!")

    #  open-set
    open_test(model)
