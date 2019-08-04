from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os 
import numpy as np
import sys
sys.path.append('.')
import torch
from torch import nn
from torch.nn import init
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import TenCrop, Lambda, Resize
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss,FocalLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from sklearn.cluster import DBSCAN,AffinityPropagation
from reid.rerank import *
# from reid.rerank_plain import *

def get_data(name, data_dir, height, width, batch_size,
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval # use all training image to train
    num_classes = dataset.num_trainval_ids

    transformer = T.Compose([
        Resize((height,width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, extfeat_loader, test_loader

def get_source_data(name, data_dir, height, width, batch_size,
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval # use all training image to train
    num_classes = dataset.num_trainval_ids

    transformer = T.Compose([
        Resize((height,width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, extfeat_loader


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)

    ## get_source_data
    src_dataset, src_extfeat_loader = \
        get_source_data(args.src_dataset, args.data_dir, args.height,
                        args.width, args.batch_size, args.workers)
    # get_target_data
    tgt_dataset, num_classes, tgt_extfeat_loader, test_loader = \
        get_data(args.tgt_dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(2048) -> FC(args.features)
    if args.src_dataset == 'dukemtmc':
        model = models.create(args.arch, num_classes=0) #duke
    elif args.src_dataset == 'market1501':
        model = models.create(args.arch, num_classes=0)
    else:
        raise RuntimeError('Please specify the number of classes (ids) of the network.')

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        print('Resuming checkpoints from finetuned model on another dataset...\n')
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise RuntimeWarning('Not using a pre-trained model')
    model = nn.DataParallel(model).cuda()
   
    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model, print_freq=args.print_freq)
    print("Test with the original model trained on source domain:")
    # best_top1 = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
    if args.evaluate:
        return

    # Criterion
    criterion = []
    criterion.append(TripletLoss(margin=args.margin,num_instances=args.num_instances).cuda())
    criterion.append(TripletLoss(margin=args.margin,num_instances=args.num_instances).cuda())

    #multi lr
    base_param_ids = set(map(id, model.module.base.parameters()))
    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
    param_groups = [
        {'params': model.module.base.parameters(), 'lr_mult': 1.0},
        {'params': new_params, 'lr_mult': 1.0}]
    # Optimizer
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=0.9, weight_decay=args.weight_decay)    

    ##### adjust lr
    def adjust_lr(epoch):
        if epoch <= 7:
            lr = args.lr
        elif epoch <=14:
            lr = 0.3 * args.lr
        else:
            lr = 0.1 * args.lr
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    ##### training stage transformer on input images
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        Resize((args.height,args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, sh=0.2, r1=0.3)
    ])

    top_percent = args.rho
    # Start training
    iter_nums = args.iteration
    cam_index = [ [] for _ in range(tgt_dataset.num_cams)]
    for i, (_, _, c) in enumerate(tgt_dataset.trainval):
        cam_index[c].append(i)
    tgt_fnames = [ f for f, _, _ in tgt_dataset.trainval]

    for iter_n in range(0, iter_nums):
        #### get source datas' feature
        source_features, _ = extract_features(model, src_extfeat_loader)
        source_features = torch.cat([source_features[f].unsqueeze(0) for f, _, _ in src_dataset.trainval], 0) # synchronization feature order with s_dataset.trainval
        #### extract training images' features
        print('Iteration {}: Extracting Target Dataset Features...'.format(iter_n+1))
        target_features, _ = extract_features(model, tgt_extfeat_loader)
        target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in tgt_dataset.trainval], 0) # synchronization feature order with dataset.trainval
        #### calculate distance and rerank result
        print('Calculating feature distances...') 
        target_features = target_features.numpy()
        # euclidean_dist, rerank_dist = re_ranking(
        #     source_features, target_features, lambda_value=0)  # lambda=1 means only source dist
        # if iter_n==0:
        #     tmp_dist = rerank_dist
        #     ####DBSCAN cluster
        #     tri_mat = np.triu(tmp_dist,1)       # tri_mat.dim=2
        #     tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
        #     tri_mat = np.sort(tri_mat,axis=None)
        #     top_num = np.round(top_percent*tri_mat.size).astype(int)
        #     eps = tri_mat[:top_num].mean()
        #     print('eps in cluster: {:.3f}'.format(eps))
        #     cluster = DBSCAN(eps=eps,min_samples=2, metric='precomputed', n_jobs=8)

        # euclidean_dist = 1.0*euclidean_dist + 0.0*rerank_dist # euclidean_dist & source_dist
        #import pdb;pdb.set_trace()
        #if iter_n%5==0:
        cross_cluster = [ [] for _ in range(tgt_dataset.num_cams) ]
        p = 0
        cluster_center = []
        for i, c in enumerate(cam_index):
            # rerank_dist_c = np.vstack([rerank_dist[m, c] for m in c])
            target_features_c = np.vstack([target_features[m] for m in c])
            _, rerank_dist_c = re_ranking(
                source_features, target_features_c, lambda_value=0)
            if iter_n==0:
                # tmp_dist = rerank_dist_c
                # ####DBSCAN cluster
                # tri_mat = np.triu(tmp_dist,1)       # tri_mat.dim=2
                # tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
                # tri_mat = np.sort(tri_mat,axis=None)
                # top_num = np.round(top_percent*tri_mat.size).astype(int)
                # eps = tri_mat[:top_num].mean()
                # print('eps in cluster: {:.3f}'.format(eps))
                # cluster = DBSCAN(eps=eps,min_samples=2, metric='precomputed', n_jobs=8)
                cluster = AffinityPropagation(max_iter=200, convergence_iter=15, affinity='precomputed')
            #### select & cluster images as training set of this epochs
            print('Clustering and labeling under single camera...')
            labels = cluster.fit_predict(rerank_dist_c)
            num_ids = len(set(labels))  ##for DBSCAN cluster
            print('Iteration {} have {} training ids'.format(iter_n+1, num_ids))
            p += num_ids
            cross_cluster[i] = [ [] for _ in range(num_ids)]
            assert num_ids == len(cluster.cluster_centers_indices_)
            cluster_center.append(target_features[np.array(c)[cluster.cluster_centers_indices_].tolist()])
            # cluster_center += np.array(c)[cluster.core_sample_indices_].tolist()
            q = 0
            print(len(cross_cluster[i]))
            print(np.max(labels))
            for j, label in enumerate(labels):
                if label == -1:
                    continue
                # print(cross_cluster[i][label])
                # print(c[j])
                q += 1
                cross_cluster[i][label].append(c[j])
            print('{}/{}'.format(q, len(labels)))
    
        cross_cluster_ = []
        cluster_features = np.vstack(cluster_center)
        print(cluster_features.shape)
        for x in cross_cluster:
            for i, y in enumerate(x):
                cross_cluster_.append(y)
        assert len(cross_cluster_) == p
        assert len(cluster_features) == p

        _, rerank_dist_cluster = re_ranking(
            source_features, cluster_features, lambda_value=0)
        if iter_n==0:
            # tmp_dist = rerank_dist_cluster
            # ####DBSCAN cluster
            # tri_mat = np.triu(tmp_dist,1)       # tri_mat.dim=2
            # tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            # tri_mat = np.sort(tri_mat,axis=None)
            # top_num = np.round(top_percent*tri_mat.size).astype(int)
            # eps = tri_mat[:top_num].mean()
            # print('eps in cluster: {:.3f}'.format(eps))
            # cluster = DBSCAN(eps=eps,min_samples=2,metric='precomputed', n_jobs=8)
            cluster = AffinityPropagation(max_iter=200, convergence_iter=15, affinity='precomputed')
        print('Clustering and labeling among different cameras...')
        labels = cluster.fit_predict(rerank_dist_cluster)
        print(len(labels))
        assert len(labels) == len(cross_cluster_)
        num_ids = len(set(labels)) - 1  ##for DBSCAN cluster
        print('Iteration {} have {} training ids'.format(iter_n+1, num_ids))
        
        del target_features
        del source_features

        #### generate new dataset
        new_dataset = []
        for i, label in enumerate(labels):
            if label == -1:
                continue
            for j in cross_cluster_[i]:
                new_dataset.append((tgt_fnames[j], label, 0))
        print('Iteration {} have {} training images'.format(iter_n+1, len(new_dataset)))
        
        train_loader = DataLoader(
            Preprocessor(new_dataset, root=tgt_dataset.images_dir,
                         transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(new_dataset, args.num_instances),
            pin_memory=True, drop_last=True)

        #### change the out_channels of last classify layer to fit new generated dataset
        #model.module.classifier_x2 = nn.Linear(2048, num_ids).cuda()
        #init.normal(model.module.classifier_x2.weight, std=0.001)
        #init.constant(model.module.classifier_x2.bias, 0)
        #### train model with new generated dataset
        top1 = iter_trainer(model, tgt_dataset, train_loader, test_loader, optimizer, criterion, args.epochs, args.logs_dir, args.print_freq)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': iter_n + 1,
            'best_top1': best_top1,
            'num_ids': num_ids,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(iter_n+1, top1, best_top1, ' *' if is_best else ''))

def iter_trainer(model, dataset, train_loader, test_loader, optimizer, criterion, epochs, logs_dir, print_freq):
    # Trainer
    best_top1 = 0
    trainer = Trainer(model, criterion)
    evaluator = Evaluator(model, print_freq=print_freq)
    # Start training
    for epoch in range(0, epochs):
        trainer.train(epoch, train_loader, optimizer)
    #evaluate
    top1 = evaluator.evaluate(test_loader, dataset.query, dataset.gallery)

    return top1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('--src-dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('--tgt-dataset', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num_instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help="balancing parameter, default: 0.1")
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    # optimizer
    parser.add_argument('--lr', type=float, default=6e-5,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, metavar='PATH',
                        default='/data/liangchen.song/models/torch/trained/dukemtmc_trained.pth.tar')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--iteration', type=int, default=30)
    parser.add_argument('--no-rerank', action='store_true', help="train without rerank")
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='./data/')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--gpu-devices', default='0,1', type=str, 
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    main(parser.parse_args())
