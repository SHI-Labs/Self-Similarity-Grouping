from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os 
import numpy as np
import sys
sys.path.append('.')
import pickle
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
from reid.trainers import Trainer, FinedTrainer, FinedTrainer2, JointTrainer2
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from sklearn.cluster import DBSCAN,AffinityPropagation
from reid.rerank import *
from reid.eug import *
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
    num_class = 0
    if args.src_dataset == 'dukemtmc':
        model = models.create(args.arch, num_classes=num_class, num_split=args.num_split, cluster=args.dce_loss) #duke
    elif args.src_dataset == 'market1501':
        model = models.create(args.arch, num_classes=num_class, num_split=args.num_split, cluster=args.dce_loss)
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
    best_top1 = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
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

    # Start training
    iter_nums = args.iteration
    start_epoch = args.start_epoch
    cluster_list = []
    top_percent = args.rho
    EF = 100 // iter_nums + 1
    eug = None
    for iter_n in range(start_epoch, iter_nums):
        #### get source datas' feature
        if args.load_dist and iter_n == 0:
            dist = pickle.load(open('dist' + str(args.num_split) + '.pkl', 'rb'))
            euclidean_dist_list = dist['euclidean']
            rerank_dist_list = dist['rerank']
        else:
            source_features, _ = extract_features(model, src_extfeat_loader, for_eval=False)
            if isinstance(source_features[src_dataset.trainval[0][0]], list):
                len_f = len(source_features[src_dataset.trainval[0][0]])
                source_features = [torch.cat([source_features[f][i].unsqueeze(0) for f, _, _ in src_dataset.trainval], 0) for i in range(len_f)]
            else:
                source_features = torch.cat([source_features[f].unsqueeze(0) for f, _, _ in src_dataset.trainval], 0) # synchronization feature order with s_dataset.trainval
            #### extract training images' features
            print('Iteration {}: Extracting Target Dataset Features...'.format(iter_n+1))
            target_features, _ = extract_features(model, tgt_extfeat_loader, for_eval=False)
            if isinstance(target_features[tgt_dataset.trainval[0][0]], list):
                len_f = len(target_features[tgt_dataset.trainval[0][0]])
                target_features = [torch.cat([target_features[f][i].unsqueeze(0) for f, _, _ in tgt_dataset.trainval], 0) for i in range(len_f)]
            else:
                target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in tgt_dataset.trainval], 0) # synchronization feature order with dataset.trainval
            #### calculate distance and rerank result
            print('Calculating feature distances...') 
            # target_features = target_features.numpy()
            euclidean_dist_list, rerank_dist_list = compute_dist(
                source_features, target_features, lambda_value=args.lambda_value, no_rerank=args.no_rerank, num_split=args.num_split) # lambda=1 means only source dist
            del target_features
            del source_features
        
        labels_list, cluster_list = generate_selflabel(
            euclidean_dist_list, rerank_dist_list, iter_n, args, cluster_list)
        #### generate new dataset
        train_loader = generate_dataloader(tgt_dataset, labels_list, train_transformer, iter_n, args)

        if iter_n == 5:
            u_data, l_data = updata_lable(tgt_dataset, labels_list[0], args.tgt_dataset, sample=args.sample)
            eug = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=num_class, 
            data_dir=args.data_dir, l_data=l_data, u_data=u_data, print_freq=args.print_freq, 
            save_path=args.logs_dir, pretrained_model=model, rerank=True)
            eug.model = model

        if eug is not None:
            nums_to_select = int(min((iter_n + 1) * int(len(u_data) // (iter_nums)), len(u_data)))
            pred_y, pred_score = eug.estimate_label()
            
            print('This is running {} with EF= {}%, step {}:\t Nums_to_be_select {}, \t Logs-dir {}'.format(
                args.mode, EF, iter_n+1, nums_to_select, args.logs_dir
            ))
            selected_idx = eug.select_top_data(pred_score, nums_to_select)
            new_train_data = eug.generate_new_train_data(selected_idx, pred_y)
            eug_dataloader = eug.get_dataloader(new_train_data, training=True)

            top1 = iter_trainer(model, tgt_dataset, train_loader, eug_dataloader, test_loader, optimizer, 
                criterion, args.epochs, args.logs_dir, args.print_freq, args.lr)
            eug.model = model
            del train_loader
            # del eug_dataloader
        else:
            top1 = iter_trainer(model, tgt_dataset, train_loader, None, test_loader, optimizer, 
            criterion, args.epochs, args.logs_dir, args.print_freq, args.lr)
            del train_loader

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': iter_n + 1,
            'best_top1': best_top1,
            # 'num_ids': num_ids,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(iter_n+1, top1, best_top1, ' *' if is_best else ''))

def iter_trainer(model, dataset, train_loader, eug_dataloader, test_loader, optimizer, 
    criterion, epochs, logs_dir, print_freq, lr):
    # Trainer
    best_top1 = 0
    # trainer = Trainer(model, criterion)
    if eug_dataloader is None:
        trainer = FinedTrainer2(model, criterion)
        evaluator = Evaluator(model, print_freq=print_freq)
        # Start training
        for epoch in range(0, epochs):
            adjust_lr(lr, epoch, optimizer)
            trainer.train(epoch, train_loader, optimizer)
    else:
        trainer = JointTrainer2(model, criterion)
        evaluator = Evaluator(model, print_freq=print_freq)
        # Start training
        for epoch in range(0, epochs):
            adjust_lr(lr, epoch, optimizer)
            trainer.train(epoch, train_loader, eug_dataloader, optimizer)
    #evaluate
    top1 = evaluator.evaluate(test_loader, dataset.query, dataset.gallery)

    return top1


def adjust_lr(init_lr, epoch, optimizer, step_size=55):
    lr = init_lr / (10 ** (epoch // step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)

    if epoch % step_size == 0:
        print("Epoch {}, current lr {}".format(epoch, lr))


def compute_dist(source_features, target_features, lambda_value, no_rerank, num_split=2):
    euclidean_dist_list = []
    rerank_dist_list = []
    if isinstance(source_features, list): 
        for (s, t) in zip(source_features, target_features):
            _, rerank_dist = re_ranking(
                s.numpy(), t.numpy(), 
                lambda_value=lambda_value, 
                no_rerank=no_rerank
            )
            rerank_dist_list.append(rerank_dist)
            euclidean_dist_list.append([])
            del rerank_dist
    else:
        _, rerank_dist = re_ranking(
            source_features.numpy(), 
            target_features.numpy(),
            lambda_value=lambda_value, no_rerank=no_rerank
        )
        rerank_dist_list.append(rerank_dist)
        euclidean_dist_list.append([])
        del rerank_dist
    return euclidean_dist_list, rerank_dist_list


def generate_selflabel(e_dist, r_dist, n_iter, args, cluster_list=[]):
    labels_list = []
    for s in range(len(r_dist)):
        if n_iter==args.start_epoch:
            if args.no_rerank:
                tmp_dist = e_dist[s]
            else:
                tmp_dist = r_dist[s]
            ####DBSCAN cluster
            tri_mat = np.triu(tmp_dist,1)       # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            top_num = np.round(args.rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps in cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps,min_samples=4, metric='precomputed', n_jobs=8)
            cluster_list.append(cluster)
        else:
            cluster = cluster_list[s]
        #### select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        if args.no_rerank:
            #euclidean_dist = -1.0 * euclidean_dist #for similarity matrix
            labels = cluster.fit_predict(e_dist[s])
        else:
            #rerank_dist = -1.0 * rerank_dist  #for similarity matrix
            labels = cluster.fit_predict(r_dist[s])
        num_ids = len(set(labels)) - 1  ##for DBSCAN cluster
        #num_ids = len(set(labels)) ##for affinity_propagation cluster
        print('Iteration {} have {} training ids'.format(n_iter+1, num_ids))
        labels_list.append(labels)
        del labels
        del cluster
    return labels_list, cluster_list

def generate_dataloader(tgt_dataset, labels_list, train_transformer, iter_n, args):
    new_dataset = []
    for i, (fname, _, _) in enumerate(tgt_dataset.trainval):
        label = []
        for s in range(len(labels_list)):
            label.append(labels_list[s][i])
        if -1 in label:
            continue
        new_dataset.append((fname, label, 0))
    print('Iteration {} have {} training images'.format(iter_n+1, len(new_dataset)))
    train_loader = DataLoader(
        Preprocessor(new_dataset, root=tgt_dataset.images_dir,
                        transform=train_transformer),
        batch_size=args.batch_size, num_workers=4,
        sampler=RandomIdentitySampler(new_dataset, args.num_instances),
        pin_memory=True, drop_last=True
    )
    return train_loader

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
    parser.add_argument('--mode', type=str,  default="Dissimilarity",
                        choices=["Classification", "Dissimilarity", "Weight"])
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
    parser.add_argument('--start-epoch', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--num-split', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=30)
    parser.add_argument('--no-rerank', action='store_true', help="train without rerank")
    parser.add_argument('--dce-loss', action='store_true', help="train without rerank")
    parser.add_argument('--sample', type=str, default='random', choices=['random', 'random_c', 'cluster', 'oneshot'])
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='./data/')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--load-dist', action='store_true', help='load pre-compute distance')
    parser.add_argument('--gpu-devices', default='0,1', type=str, 
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    main(parser.parse_args())
