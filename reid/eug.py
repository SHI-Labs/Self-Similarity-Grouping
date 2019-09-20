import torch
from torch import nn
from reid import models
from reid.trainers import Trainer, DistillTrainer
from reid.evaluators import extract_features, Evaluator
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss,FocalLoss
import numpy as np
from collections import OrderedDict
import os.path as osp
import pickle
from reid.utils.serialization import load_checkpoint
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.rerank_initial import re_ranking_init
import random
from copy import deepcopy


class EUG():
    def __init__(self, model_name, batch_size, mode, num_classes, data_dir, l_data, u_data, save_path, print_freq, 
        dropout=0.5, pretrained_model=None, triplet=False, rerank=False):

        self.model_name = model_name
        self.num_classes = num_classes
        self.mode = mode
        self.data_dir = data_dir
        self.save_path = save_path

        self.l_data = l_data
        self.l_data = [[f, l, 1.0] for f, l, _ in l_data]
        self.u_data = u_data
        self.l_label = np.array([label for _,label,_ in l_data])
        self.u_label = np.array([label for _,label,_ in u_data])


        self.dataloader_params = {}
        self.dataloader_params['height'] = 256
        self.dataloader_params['width'] = 128
        self.dataloader_params['batch_size'] = batch_size
        self.dataloader_params['workers'] = 6


        self.batch_size = batch_size
        self.data_height = 256
        self.data_width = 128
        self.data_workers = 6

        self.eval_bs = batch_size
        self.dropout = dropout
        # self.output_feature = output_feature
        self.model = pretrained_model
        if self.model and num_classes > 0:
            self.model_distill = deepcopy(self.model)
        self.print_freq = print_freq
        self.num_instances = 4
        self.rerank = rerank


    def get_dataloader(self, dataset, training=False) :
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        if training:
            transformer = T.Compose([
                T.RandomSizedRectCrop(self.data_height, self.data_width),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalizer,
                T.RandomErasing(probability=0.5, sh=0.2, r1=0.3)
            ])

        else:
            transformer = T.Compose([
                T.Resize((self.data_height, self.data_width)),
                T.ToTensor(),
                normalizer,
            ])
        if training and self.num_classes == 0:
            data_loader = DataLoader(
                Preprocessor(dataset, root='', transform=transformer),
                batch_size=self.batch_size, num_workers=self.data_workers,
                sampler=RandomIdentitySampler(dataset, self.num_instances),
                pin_memory=True, drop_last=training)
        else:
            data_loader = DataLoader(
                Preprocessor(dataset, root='', transform=transformer),
                batch_size=self.batch_size, num_workers=self.data_workers,
                shuffle=training, pin_memory=True, drop_last=training)


        current_status = "Training" if training else "Test"
        print("create dataloader for {} with batch_size {}".format(current_status, self.batch_size))
        return data_loader




    def train(self, train_data, step, epochs=70, step_size=55, init_lr=0.1, dropout=0.5):

        """ create model and dataloader """
        if self.model:
            model = self.model
        else:
            model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes, mode=self.mode)
            model = nn.DataParallel(model).cuda()
        dataloader = self.get_dataloader(train_data, training=True)


        # the base parameters for the backbone (e.g. ResNet50)
        base_param_ids = set(map(id, model.module.base.parameters()))

        # we fixed the first three blocks to save GPU memory
        # base_params_need_for_grad = filter(lambda p: p.requires_grad, model.module.base.parameters())

        # params of the new layers
        new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
        # set the learning rate for backbone to be 0.1 times
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 1.0},
            # {'params': base_params_need_for_grad, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

        criterion = []
        if self.num_classes == 0:
            criterion.append(TripletLoss(margin=0.3, num_instances=self.num_instances).cuda())
            criterion.append(TripletLoss(margin=0.3, num_instances=self.num_instances).cuda())
            trainer = Trainer(model, criterion)
        else:
            criterion.append(nn.CrossEntropyLoss().cuda())
            criterion.append(nn.MSELoss().cuda())
            trainer = DistillTrainer(model, self.model_distill, criterion)
        optimizer = torch.optim.SGD(param_groups, lr=init_lr, momentum=0.9, weight_decay = 5e-4, nesterov=True)

        # change the learning rate by step
        def adjust_lr(epoch, step_size):
            lr = init_lr / (10 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

            if epoch % step_size == 0:
                print("Epoch {}, current lr {}".format(epoch, lr))
        # def adjust_lr(epoch):
        #     if epoch <=7:
        #         lr = args.lr
        #     elif epoch <= 14:
        #         lr = 0.3 * args.lr
        #     else:
        #         lr = 0.1 * args.lr
        #     for g in optimizer.param_groups:
        #         g['lr'] = lr * g.get('lr_mult', 1)

        """ main training process """
        for epoch in range(epochs):
            adjust_lr(epoch, step_size)
            trainer.train(epoch, dataloader, optimizer, print_freq=20)
        self.model = model


    def get_feature(self, dataset):
        dataloader = self.get_dataloader(dataset, training=False)
        features,_ = extract_features(self.model, dataloader)
        features = np.array([logit.numpy() for logit in features.values()])
        return features

    def get_Classification_result(self):
        logits = self.get_feature(self.u_data)
        exp_logits = np.exp(logits)
        predict_prob = exp_logits / np.sum(exp_logits,axis=1).reshape((-1,1))
        assert len(logits) == len(predict_prob)
        assert predict_prob.shape[1] == self.num_classes

        pred_label = np.argmax(predict_prob, axis=1)
        pred_score = predict_prob.max(axis=1)
        print("get_Classification_result", predict_prob.shape)


        num_correct_pred = 0
        for idx, p_label in enumerate(pred_label):
            if self.u_label[idx] == p_label:
                num_correct_pred +=1


        print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
            self.mode, num_correct_pred, pred_label.shape[0], num_correct_pred/pred_label.shape[0]))

        return pred_label, pred_score



    def get_Dissimilarity_result(self, weight=False):

        # extract feature 
        u_feas = self.get_feature(self.u_data)
        l_feas = self.get_feature(self.l_data)
        print("u_features", u_feas.shape, "l_features", l_feas.shape)

        if not self.rerank:
            scores = np.zeros((u_feas.shape[0]))
            labels = np.zeros((u_feas.shape[0]))

            num_correct_pred = 0
            for idx, u_fea in enumerate(u_feas):
                diffs = l_feas - u_fea
                dist = np.linalg.norm(diffs,axis=1)
                index_min = np.argmin(dist)
                scores[idx] = -dist[index_min]
                labels[idx] = int(self.l_label[index_min])
                # count the correct number of Nearest Neighbor prediction
                if self.u_label[idx] == int(self.l_label[index_min]):
                    num_correct_pred +=1

            print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
                self.mode, num_correct_pred, u_feas.shape[0], num_correct_pred/u_feas.shape[0]))

            del u_feas
            del l_feas
            
            return labels, scores
        else:
            u_l_dist = np.dot(u_feas, np.transpose(l_feas))
            u_u_dist = np.dot(u_feas, np.transpose(u_feas))
            l_l_dist = np.dot(l_feas, np.transpose(l_feas))
            re_rank_dist = re_ranking_init(u_l_dist, u_u_dist, l_l_dist)

            scores = np.zeros((u_feas.shape[0]))
            labels = np.zeros((u_feas.shape[0]))
            confidence = np.zeros((u_feas.shape[0]))
            num_correct_pred = 0
            for idx, dist in enumerate(re_rank_dist):
                index_min = np.argmin(dist)
                scores[idx] = -dist[index_min]
                labels[idx] = int(self.l_label[index_min])
                confidence[idx] = 1 - dist[index_min] / np.max(re_rank_dist[:, index_min])
                # count the correct number of Nearest Neighbor prediction
                if self.u_label[idx] == int(self.l_label[index_min]):
                    num_correct_pred +=1

            print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
                self.mode, num_correct_pred, u_feas.shape[0], num_correct_pred/u_feas.shape[0]))

            del u_feas
            del l_feas
            del u_l_dist
            del u_u_dist
            del l_l_dist
            del re_rank_dist
            if weight:
                return labels, scores, confidence
            else:
                return labels, scores

    def estimate_label(self):

        print("label estimation by {} mode.".format(self.mode))

        if self.mode == "Dissimilarity": 
            # predict label by dissimilarity cost
            [pred_label, pred_score] = self.get_Dissimilarity_result()
            return pred_label, pred_score
        elif self.mode == "Classification": 
            # predict label by classification
            [pred_label, pred_score] = self.get_Classification_result()
            return pred_label, pred_score
        elif self.mode == 'Weight':
            [pred_label, pred_score, confidence] = self.get_Dissimilarity_result(True)
            return pred_label, pred_score, confidence
        else:
            raise ValueError

        

    def select_top_true_data(self, pred_label, pred_score, nums_to_select):
        v = np.zeros(len(pred_score))
        index = np.argsort(-pred_score)
        for i in range(nums_to_select):
            if pred_label[index[i]] != -1:
                v[index[i]] = 1
        return v.astype('bool')
    
    def select_top_data(self, pred_score, nums_to_select):
        v = np.zeros(len(pred_score))
        index = np.argsort(-pred_score)
        for i in range(nums_to_select):
            v[index[i]] = 1
        return v.astype('bool')


    def generate_new_train_data(self, sel_idx, pred_y):
        """ generate the next training data """

        seletcted_data = []
        correct, total = 0, 0
        for i, flag in enumerate(sel_idx):
            if flag: # if selected
                seletcted_data.append([self.u_data[i][0], int(pred_y[i]), self.u_data[i][2]])
                total += 1
                if self.u_label[i] == int(pred_y[i]):
                    correct += 1
        acc = correct / total

        new_train_data = self.l_data + seletcted_data
        print("selected pseudo-labeled data: {} of {} is correct, accuracy: {:0.4f}  new train data: {}".format(
                correct, len(seletcted_data), acc, len(new_train_data)))

        return new_train_data


    def resume(self, ckpt_file, step):
        print("continued from step", step)
        model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes, mode=self.mode)
        # self.model = nn.DataParallel(model).cuda()
        model.load_state_dict(load_checkpoint(ckpt_file), strict=False)
        model_distill = deepcopy(model)
        # model.load_state_dict(load_checkpoint_new(ckpt_file), strict=False)
        self.model = nn.DataParallel(model).cuda()
        self.model_distill = nn.DataParallel(model_distill).cuda()


"""
    Get one-shot split for the input dataset.
"""
def updata_lable(dataset, label, name, sample='random', load_path='random_split/', seed=0):
    np.random.seed(seed)
    random.seed(seed)
    load_path = load_path + sample + '_' + name + '.pkl'
    if osp.exists(load_path):
        with open(load_path, "rb") as fp:
            dataset = pickle.load(fp)
            label_dataset = dataset["label set"]
            unlabel_dataset = dataset["unlabel set"]

        print("  labeled  |   N/A | {:8d}".format(len(label_dataset)))
        print("  unlabel  |   N/A | {:8d}".format(len(unlabel_dataset)))
        print("\nLoad one-shot split from", load_path)
        return unlabel_dataset, label_dataset
    
    print("Randomly Create new one-shot split and save it to", load_path)
    if sample == 'random':
        num_ids = len(set(label)) - 1
        label_dataset = []
        unlabel_dataset = []
        # select_index = np.random.choice(np.arange(len(dataset.trainval)), num_ids, replace=True)
        label_dataset = [[osp.join(dataset.images_dir, f), pid, camid] for  f, pid, camid in dataset.trainval]
        np.random.shuffle(label_dataset)
        label_dataset = label_dataset[:num_ids]
        labeled_imgIDs = [fname for _, (fname, _, _) in enumerate(label_dataset)]
        for (fname, pid, camid) in dataset.trainval:
            fname = osp.join(dataset.images_dir, fname)
            if fname not in labeled_imgIDs:
                unlabel_dataset.append([fname, pid, camid])
        print("  labeled    | N/A | {:8d}".format(len(label_dataset)))
        print("  unlabeled  | N/A | {:8d}".format(len(unlabel_dataset)))
    elif sample == 'cluster':
        label_dataset = []
        unlabel_dataset = []
        data_key = {}
        cluster = {}
        for i, (f, pid, camid) in enumerate(dataset.trainval):
            if label[i] == -1:
                continue
            if label[i] not in list(cluster.keys()):
                cluster[label[i]] = []
            cluster[label[i]].append([osp.join(dataset.images_dir, f), pid, camid])
        for i in list(cluster.keys()):
            f = cluster[i]
            np.random.shuffle(f)
            label_dataset.append(f[0])
            # label_dataset.append(f[1])
        labeled_imgIDs = [fname for _, (fname, _, _) in enumerate(label_dataset)]
        for (fname, pid, camid) in dataset.trainval:
            fname = osp.join(dataset.images_dir, fname)
            if fname not in labeled_imgIDs:
                unlabel_dataset.append([fname, pid, camid])
        print("  labeled    | N/A | {:8d}".format(len(label_dataset)))
        print("  unlabeled  | N/A | {:8d}".format(len(unlabel_dataset)))
    with open(load_path, "wb") as fp:
        pickle.dump({"label set": label_dataset, "unlabel set":unlabel_dataset}, fp)
    return unlabel_dataset, label_dataset




