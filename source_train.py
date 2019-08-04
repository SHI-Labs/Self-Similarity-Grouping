from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os 
import numpy as np
import time
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.evaluation_metrics import accuracy
from reid.utils.meters import AverageMeter
from reid.eug import *

class Trainer(object):
    def __init__(self, model, criterions, print_freq=1):
        super(Trainer, self).__init__()
        self.model = model
        self.criterions = criterions
        self.print_freq = print_freq

    def train(self, epoch, data_loader, optimizer):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets, epoch)
            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs)
        #new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[0], targets, epoch)
        # x2 global feature cross entropy loss
        loss_global = self.criterions[1](outputs[1], targets)
        prec_global, = accuracy(outputs[1].data, targets.data)
        prec_global = prec_global[0]

        return loss_tri+loss_global, prec_global


from torch.nn import functional as F
class ResNet(models.ResNet):
    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x1 = F.avg_pool2d(x, x.size()[2:])
        x1 = x1.view(x1.size(0), -1)
        x2 = self.feat(x1)
        x2 = self.feat_bn(x2)
        x2 = self.relu(x2)
        x2 = self.drop(x2)
        x2 = self.classifier_x2(x2)
        return x1, x2

def get_data(name, data_dir, height, width, batch_size,
             workers, num_select=1):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval # use all training image to train
    num_classes = dataset.num_trainval_ids
    
    if num_select == 1:
        load_path = './oneshot_split/oneshot_'
    elif num_select == 3:
        load_path = './oneshot_split/threeshot_'
    elif num_select == 5:
        load_path = './oneshot_split/fiveshot_'
    elif num_select == 8:
        load_path = './oneshot_split/eightshot_'
    else:
        raise RuntimeWarning('Please choose from 1,3,5,8')
    
    load_path = load_path + name + '.pkl'
    
    l_data, u_data = get_one_shot_in_cam1(dataset, load_path, num_select)

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

    return dataset, num_classes, extfeat_loader, test_loader, l_data, u_data



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

    dataset, num_classes, _, test_loader, l_data, u_data = \
        get_data(args.dataset, args.data_dir, args.height,
                args.width, args.batch_size, args.workers, args.num_select)
    if not args.evaluate:
        if args.dataset == 'dukemtmc':
            model = models.create(args.arch, num_classes=0)
        elif args.dataset == 'market1501':
            model = models.create(args.arch, num_classes=0)
        elif args.dataset == 'msmt17':
            model = models.create(args.arch, num_classes=0)
        else:
            raise RuntimeError("Please specify the number of classes(ids) of the network")
        # Load from checkpoint
        start_epoch = best_top1 = 0
        if args.resume:
            print('Resuming checkpoints from finetuned model on another dataset...\n')
            checkpoint = load_checkpoint(args.resume)
            model.load_state_dict(checkpoint, strict=False)
        else:
            raise RuntimeWarning('Not using a pre-trained model')
        model = nn.DataParallel(model).cuda()

        eug = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=0, 
                data_dir=args.data_dir, l_data=l_data, u_data=u_data, print_freq=args.print_freq, save_path=args.logs_dir, pretrained_model=model, rerank=True)
        # Distance metric
        metric = DistanceMetric(algorithm=args.dist_metric)

        nums_to_select = len(u_data)
        pred_y, pred_score = eug.estimate_label()
        selected_idx = eug.select_top_data(pred_score, nums_to_select)
        new_train_data = eug.generate_new_train_data(selected_idx, pred_y)
        eug_dataloader = eug.get_dataloader(new_train_data, training=True)
        del model

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    model = ResNet(int(args.arch[-2:]), pretrained=True, num_classes=num_classes)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume_target:
        checkpoint = torch.load(args.resume_target)
        model.load_state_dict(checkpoint['state_dict'])
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()
   
    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model, args.print_freq)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        return

    # Criterion
    criterion = []
    criterion.append(TripletLoss(args.margin, args.num_instances, False).cuda())
    criterion.append(nn.CrossEntropyLoss().cuda())

    #multi lr
    base_param_ids = set(map(id, model.module.base.parameters()))
    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
    param_groups = [
        {'params': model.module.base.parameters(), 'lr_mult': 1.0},
        {'params': new_params, 'lr_mult': 1.0}]
    # Optimizer
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, 
                                 weight_decay=args.weight_decay)

    # Trainer
    trainer = Trainer(model, criterion, args.print_freq)
    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 60 else \
            args.lr * (0.1 ** (epoch // 60))
            # args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, eug_dataloader, optimizer)
        if epoch < args.start_save:
            continue
        if (epoch % 5 == 0 and epoch > 0) or epoch == args.epochs:
            top1 = evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = torch.load(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--mode', type=str,  default="Dissimilarity",
                        choices=["Classification", "Dissimilarity"])
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0003,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--resume-target', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start-save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--num-select', type=int, default=1)
    # metric learning
    parser.add_argument('--dist_metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--gpu-devices', default='0,1', type=str, 
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    main(parser.parse_args())