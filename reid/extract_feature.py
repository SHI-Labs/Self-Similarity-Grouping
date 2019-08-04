from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import os
import time
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import TenCrop, Resize, Lambda, CenterCrop

from reid.feature_extraction import extract_cnn_feature
from reid import datasets
from reid import models
from reid.utils.meters import AverageMeter
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from test_dataset import TestDataset
from sklearn.preprocessing import normalize


GT_FILES = './data/market1501/raw/Market-1501-v15.09.15/gt_bbox'

def get_dataloader(batch_size=64, workers=4):
    #prepare dataset
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


    test_transformer = T.Compose([
        Resize((256, 128)),
        T.ToTensor(),
        normalizer,
    ])

    #test_time augmentation
    #test_transformer = T.Compose([
    #        Resize((288, 144)),
	#    TenCrop((256,128)),
    #        Lambda(lambda crops: torch.stack([normalizer(T.ToTensor()(crop)) for crop in crops])),
    #        ])

    single_train_dir = './data/market1501/raw/Market-1501-v15.09.15/bounding_box_train'
    single_query_dir = './data/market1501/raw/Market-1501-v15.09.15/query'
    multi_query_dir = './data/market1501/raw/Market-1501-v15.09.15/gt_bbox'
    gallery_dir = './data/market1501/raw/Market-1501-v15.09.15/bounding_box_test'

    single_train_loader = DataLoader(TestDataset(root=single_train_dir, transform=test_transformer),
                                        batch_size=batch_size,
                                        num_workers=workers,
                                        shuffle=False, pin_memory=True)
    single_query_loader = DataLoader(TestDataset(root=single_query_dir, transform=test_transformer),
                                        batch_size=batch_size,
                                        num_workers=workers,
                                        shuffle=False, pin_memory=True)

    multi_query_loader = DataLoader(TestDataset(root=multi_query_dir, transform=test_transformer),
                                        batch_size=batch_size,
                                        num_workers=workers,
                                        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(TestDataset(root=gallery_dir, transform=test_transformer),
                                        batch_size=batch_size,
                                        num_workers=workers,
                                        shuffle=False, pin_memory=True)
    return single_train_loader, single_query_loader, multi_query_loader, gallery_loader

def extract_features(model, data_loader, print_freq=1, save_name='feature.mat'):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    ids = []
    cams = []
    features = []
    query_files = []
    end = time.time()
    for i, (imgs, fnames) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        #for test time augmentation
        #bs, ncrops, c, h, w = imgs.size()
        #outputs = extract_cnn_feature(model, imgs.view(-1,c,h,w))
        #outputs = outputs.view(bs,ncrops,-1).mean(1)
        for fname, output in zip(fnames, outputs):
            if fname[0]=='-':
                ids.append(-1)
                cams.append(int(fname[4]))
            else:
                ids.append(int(fname[:4]))
                cams.append(int(fname[6]))
            features.append(output.numpy())
            query_files.append(fname)
            batch_time.update(time.time() - end)
            end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, ids, cams, query_files

def evaluate():
    print ('Get dataloader... ')
    single_train_loader, single_query_loader, multi_query_loader, gallery_loader = get_dataloader()

    print ('Create and load pre-trained model...')
    model = models.create('resnet50',dropout=0.0, num_features=2048, num_classes=632)
    # checkpoint = load_checkpoint('./logs/deep-person-1-new-augmentation/market1501-resnet50/model_best.pth.tar')
    checkpoint = load_checkpoint('./logs/duke2market/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(model).cuda()
    model.eval()

    print ('Extract train&single_query&gallery feature...')
    single_train_feat, single_train_ids, single_train_cams, train_files = extract_features(model, single_train_loader)
    single_query_feat, single_query_ids, single_query_cams, query_files = extract_features(model, single_query_loader)
    gallery_feat, gallery_ids, gallery_cams, _ = extract_features(model, gallery_loader)

    print ('Get multi_query feature...')
    multi_query_dict = dict()
    for i, (imgs, fnames) in enumerate(multi_query_loader):
        outputs = extract_cnn_feature(model, imgs)
    	# test time augmentation
        for fname, output in zip(fnames, outputs):
            if multi_query_dict.get(fname[:7])==None:
                multi_query_dict[fname[:7]]=[]
            multi_query_dict[fname[:7]].append(output.numpy())

    query_max_feat = []
    query_avg_feat = []
    for query_file in query_files:
        index = query_file[:7]
        multi_features = multi_query_dict[index]
        multi_features = normalize(multi_features)
        query_max_feat.append(np.max(multi_features,axis=0))
        query_avg_feat.append(np.mean(multi_features,axis=0))


    assert len(query_max_feat)==len(query_avg_feat)==len(single_query_feat)

    print ('Write to mat file...')
    import scipy.io as sio
    if not os.path.exists('./matdata'):
        os.mkdir('./matdata')
    sio.savemat('./matdata/trainID.mat', {'trainID':np.array(single_train_ids)})
    sio.savemat('./matdata/queryID.mat', {'queryID':np.array(single_query_ids)})
    sio.savemat('./matdata/trainCAM.mat', {'trainCAM':np.array(single_train_cams)})
    sio.savemat('./matdata/queryCAM.mat', {'queryCAM':np.array(single_query_cams)})
    sio.savemat('./matdata/testID.mat', {'testID':np.array(gallery_ids)})
    sio.savemat('./matdata/testCAM.mat', {'testCAM':np.array(gallery_cams)})
    sio.savemat('./matdata/Hist_train.mat', {'Hist_train':np.array(single_train_feat)})
    sio.savemat('./matdata/Hist_query.mat', {'Hist_query':np.array(single_query_feat)})
    sio.savemat('./matdata/Hist_test.mat', {'Hist_test':np.array(gallery_feat)})
    sio.savemat('./matdata/Hist_query_max.mat', {'Hist_max':np.array(query_max_feat)})
    sio.savemat('./matdata/Hist_query_avg.mat', {'Hist_avg':np.array(query_avg_feat)})

    return

evaluate()
