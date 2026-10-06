import os
import argparse
import time
import torch
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader

from models.dataset import MyDataset
from models.ccnetvq import ccnetvq
from models.resnetvq import ResNetVq
# from models.co3netvq import co3netvq
from models.sfnet import SFNet
from models.resnetDHN import ResNetDHN,Bottleneck,BasicBlock
from models.resDHNvq import ResNetDHNvq
from models.compnet import compnet
from utils import *
from models.compnetvq import compnetvq
from models.ccnetvq import ccnetvq
from models.co3netvq import co3netvq
import sys


def open_test(model):

    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    open_path_rst = path_rst + "open/"

    path_hard = os.path.join(open_path_rst, 'rank1_hard')

    # train_set_file = './data/train_IITD.txt'
    # test_set_file = './data/test_IITD.txt'

    galleryset = MyDataset(txt=gallery_set_file, transforms=None, train=False)
    queryset = MyDataset(txt=query_set_file, transforms=None, train=False)

    batch_size = 128  # 128

    data_loader_gallery = DataLoader(dataset=galleryset, batch_size=batch_size, num_workers=2)
    data_loader_query = DataLoader(dataset=queryset, batch_size=batch_size, num_workers=2)

    fileDB_train = getFileNames(gallery_set_file)
    fileDB_test = getFileNames(query_set_file)

    # output dir
    if not os.path.exists(open_path_rst):
        os.makedirs(open_path_rst)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    net = model

    net.cuda()
    net.eval()

    # feature extraction:

    # start_time = time.time()

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
    # end_time = time.time()
    # print("feature extraction time cost:", end_time-start_time)

    print('\nFeature Extraction Done!')

    print('start feature matching ...\n')

    print('Verification EER of the test-test set ...')

    print('Start EER for Test-Test Set! \n')

    # verification EER of the test set
    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]

        for j in range(ntrain):
            feat2 = featDB_train[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    if not os.path.exists(open_path_rst+'veriEER'):
        os.makedirs(open_path_rst+'veriEER')
    if not os.path.exists(open_path_rst+'veriEER/rank1_hard/'):
        os.makedirs(open_path_rst+'veriEER/rank1_hard/')

    with open(open_path_rst+'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + open_path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + open_path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    # rank-1 acc
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]

        dis = np.zeros((ntrain, 1))

        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1

        idx = np.argmin(dis[:])

        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1
        else:
            testname = fileDB_test[i]
            trainname = fileDB_train[idx]
            # store similar inter-class samples
            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)
            img = np.concatenate((im_test, im_train), axis=1)
            cv.imwrite(open_path_rst + 'veriEER/rank1_hard/%6.4f_%s_%s.png' % (
                np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(open_path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)

    print('\n\nReal EER of the test set...')
    # dataset EER of the test set (the gallery set is not used)
    s = []  # matching score
    l = []  # genuine / impostor matching
    n = featDB_test.shape[0]
    for i in range(n - 1):
        feat1 = featDB_test[i]

        for jj in range(n - i - 1):
            j = i + jj + 1
            feat2 = featDB_test[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_test[j]:
                l.append(1)
            else:
                l.append(-1)

    print('feature extraction about real EER done!\n')

    with open(open_path_rst + 'veriEER/scores_EER_test.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + open_path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')
    os.system('python ./getEER.py' + '  ' + open_path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')


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
    parser.add_argument("--com_weight",type=float,default=0.8)
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
    parser.add_argument("--checkpoint_rest", type=str, default='./results/checkpoint/')
    parser.add_argument("--path_rst", type=str, default='./results/rst_test/')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    num_classes = args.id_num
    comp_weight = args.com_weight
    path_rst = args.path_rst
    gallery_set_file = args.gallery_set_file
    query_set_file = args.query_set_file
    loss_ce = args.loss_ce
    loss_tl = args.loss_tl
    vit_floor_num = args.vit_floor_num

    print("\n========== Test Config ==========")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Gallery Set: {args.gallery_set_file}")
    print(f"Query Set:   {args.query_set_file}")
    print(f"Result Path: {args.path_rst}")
    print("=================================\n")

    # 初始化模型
    # model = compnetvq(num_classes=num_classes)
    # model = ccnetvq(num_classes=num_classes, weight=comp_weight)
    model = co3netvq(num_classes=num_classes)
    # checkpoint = torch.load(args.checkpoint, map_location='cuda')
    # 2. 读取 checkpoint
    sd_vq = torch.load(args.checkpoint, map_location='cuda')
    sd_rest = torch.load(args.checkpoint_rest, map_location='cuda')

    # 兼容 {"state_dict": ...}
    sd_vq = sd_vq["state_dict"] if "state_dict" in sd_vq else sd_vq
    sd_rest = sd_rest["state_dict"] if "state_dict" in sd_rest else sd_rest

    # 3. 定义量化器参数前缀
    vq_prefixes = (
        "quantizer_cb1.",
        "quantizer_cb2.",
        "quantizer_cb3.",
    )

    # 4. 合并 state_dict
    merged_state = {}

    # 4.1 先放非量化器参数
    for k, v in sd_rest.items():
        if not k.startswith(vq_prefixes):
            merged_state[k] = v

    # 4.2 再覆盖量化器参数
    for k, v in sd_vq.items():
        if k.startswith(vq_prefixes):
            merged_state[k] = v

    # 5. 加载
    missing, unexpected = model.load_state_dict(merged_state, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    print("Loaded checkpoint successfully!")

    # 执行 open-set 测试
    open_test(model)
