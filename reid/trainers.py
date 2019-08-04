from __future__ import print_function, absolute_import
import time

import torch
from torch import nn
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterions, model_distill=None):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.model_distill = model_distill
        self.criterions = criterions

    def train(self, epoch, data_loader, optimizer, print_freq=5):
        self.model.train()
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = False

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
            #add gradient clip for lstm
            for param in self.model.parameters():
                try:
                    param.grad.data.clamp(-1., 1.)
                except:
                    continue

            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
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
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs) #outputs=[x1,x2,x3]
        #new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[0], targets, epoch)
        # x2 global feature cross entropy loss
        #loss_global = self.criterions[1](outputs[1], targets)
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        #prec_global, = accuracy(outputs[1].data, targets.data)
        #prec_global = prec_global[0]

        return loss_tri+loss_global, prec_global
        

class DistillTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        pids = pids.cuda()
        return inputs, pids
    def _forward(self, inputs, targets, epoch):
        feats, outputs = self.model(inputs)
        feats_distill, _ = self.model_distill(inputs)
        # assert feats_distill.requires_grad is False
        assert feats_distill.size(1) == 2048
        if isinstance(self.criterions[0], torch.nn.CrossEntropyLoss):
            loss_c = self.criterions[0](outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterions[0], TripletLoss):
            loss, prec = self.criterions[0](outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion[0])
        loss_d = self.criterions[1](feats, feats_distill)
        # print("Classification Loss Distillation Loss: {}/{}".format(loss_c, loss_d))
        loss = loss_c + 0.1 * loss_d
        return loss, prec



class FinedTrainer(object):
    def __init__(self, model, criterions, beta=0.5):
        super(FinedTrainer, self).__init__()
        self.model = model
        self.criterions = criterions
        self.beta = beta

    def train(self, epoch, train_loader_list, optimizer, print_freq=10):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        train_loader = train_loader_list[0]
        input_iter = [iter(x) for x in train_loader_list[1:]]

        end = time.time()

        for i, inputs in enumerate(train_loader):
            data_time.update(time.time() - end)
            inputs_p = []
            # for j, x in enumerate(input_iter):
            #     try:
            #         inputs_p.append(next(x))
            #     except:
            #         x = iter(train_loader_list[j+1])
            #         inputs_p.append(next(x))
            try:
                input_p = [next(x) for x in input_iter]
            except:
                input_iter = [iter(x) for x in train_loader_list[1:]]
                input_p = [next(x) for x in input_iter]

            # inputs, pids, _ = self._parse_data(inputs)
            # inputs_, pids_, _ = self._parse_data(inputs_)

            loss, prec = self._forward(inputs, inputs_p, epoch)
            losses.update(loss.item(), inputs[0].size(0))
            precisions.update(prec, inputs[0].size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, w = inputs
        inputs = [imgs]
        targets = pids.cuda()
        w = w.float().cuda()
        return inputs, targets, w

    def _forward(self, inputs_g, inputs_p, epoch):
        inputs, pids, _ = self._parse_data(inputs_g)
        outputs = self.model(*inputs)
        loss_global, prec_global = self.criterions[1](outputs[1], pids, epoch)
        loss_tri, prec_tri = self.criterions[0](outputs[0][0], pids, epoch)
        loss = loss_global + loss_tri
        for i, inputs in enumerate(inputs_p):
            inputs, pids, _ = self._parse_data(inputs)
            outputs = self.model(*inputs)
            loss_tri, _ = self.criterions[0](outputs[0][i+1], pids, epoch)
            loss + loss_tri
        prec = prec_global
        return loss, prec






class FinedTrainer2(object):
    def __init__(self, model, criterions, beta=0.5):
        super(FinedTrainer2, self).__init__()
        self.model = model
        self.criterions = criterions
        self.beta = beta

    def train(self, epoch, train_loader, optimizer, print_freq=10):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(train_loader):
            data_time.update(time.time() - end)

            inputs, pids, _ = self._parse_data(inputs)
            loss, prec = self._forward(inputs, pids, epoch)

            losses.update(loss.item(), inputs[0].size(0))
            precisions.update(prec, inputs[0].size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, w = inputs
        inputs = [imgs]
        targets = [p.cuda() for p in pids]
        w = w.float().cuda()
        return inputs, targets, w

    def _forward(self, inputs, pids, epoch):
        loss = 0.0
        outputs = self.model(*inputs)
        loss_global, prec_global = self.criterions[1](outputs[1], pids[0], epoch)
        loss = loss_global
        prec = prec_global
        # outputs_split = [outputs[0][0], torch.cat(outputs[0][1:], dim=1)]
        if isinstance(outputs[0], list):
            for i, output_p in enumerate(outputs[0]):
                loss_tri, prec_tri = self.criterions[0](output_p, pids[i], epoch)
                loss += loss_tri
        else:
            loss_tri, prec_tri = self.criterions[0](outputs[0], pids[0], epoch)
            loss += loss_tri
        if len(outputs) == 3:
            if isinstance(outputs[2], list):
                for out in outputs[2]:
                    target_p = self.target_distribution(out)
                    loss_function = nn.KLDivLoss(size_average=False)
                    loss += loss_function(out.log(), target_p) / out.shape[0]
                    
            else:
                target_p = self.target_distribution(outputs[2])
                loss_function = nn.KLDivLoss(size_average=False)
                loss += 3 * (loss_function(outputs[2].log(), target_p) / outputs[2].shape[0])
            return loss, prec
        else:
            prec = prec_global
            return loss, prec

    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()



class JointTrainer(object):
    def __init__(self, model, criterions, beta=0.5):
        super(JointTrainer, self).__init__()
        self.model = model
        self.criterions = criterions
        self.beta = beta

    def train(self, epoch, train_loader, eug_loader, optimizer, print_freq=10):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        eug_loader_iter = iter(eug_loader)

        end = time.time()
        for i, inputs in enumerate(train_loader):
            data_time.update(time.time() - end)
            # load source_triplet and target
            try:
                inputs_eug = next(eug_loader_iter)
            except:
                eug_loader_iter = iter(eug_loader)
                inputs_eug = next(eug_loader_iter)
            
            inputs, pids, _ = self._parse_data(inputs)
            inputs_eug, pids_eug, w_eug = self._parse_data(inputs_eug)
            # print(inputs_eug[0].size())
            # print(w_eug.size())
            # print(torch.sum(w_eug) / inputs[0].size(0))

            loss, prec = self._forward(inputs, pids, inputs_eug, pids_eug, epoch, w_eug)

            losses.update(loss.item(), pids.size(0))
            precisions.update(prec, pids.size(0) + pids_eug.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, w = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        w = w.float().cuda()
        return inputs, targets, w

    def _forward(self, inputs, pids, inputs_eug, pids_eug, epoch, w_eug=None):
        outputs = self.model(*inputs)
        outputs_eug = self.model(*inputs_eug)

        loss_tri, prec_tri = self.criterions[0](outputs[0], pids, epoch)
        loss_global, prec_global = self.criterions[1](outputs[1], pids, epoch)

        loss_tri_eug, prec_tri_eug = self.criterions[0](outputs_eug[0], pids_eug, epoch, w_eug)
        loss_global_eug, prec_global_eug = self.criterions[1](outputs_eug[1], pids_eug, epoch, w_eug)

        loss = loss_tri + loss_global + (loss_tri_eug + loss_global_eug) * ( torch.sum(w_eug) / outputs[0].size(0) )
        # loss = loss_tri + loss_global + loss_tri_eug + loss_global_eug
        prec = prec_global + prec_global_eug

        return loss, prec



class JointTrainer2(object):
    def __init__(self, model, criterions, beta=0.5):
        super(JointTrainer2, self).__init__()
        self.model = model
        self.criterions = criterions
        self.beta = beta

    def train(self, epoch, train_loader, eug_loader, optimizer, print_freq=10):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        eug_loader_iter = iter(eug_loader)

        end = time.time()
        for i, inputs in enumerate(train_loader):
            data_time.update(time.time() - end)
            # load source_triplet and target
            try:
                inputs_eug = next(eug_loader_iter)
            except:
                eug_loader_iter = iter(eug_loader)
                inputs_eug = next(eug_loader_iter)
            
            inputs, pids, _ = self._parse_data(inputs)
            inputs_eug, pids_eug, _ = self._parse_data(inputs_eug)

            loss, prec = self._forward(inputs, pids, inputs_eug, pids_eug, epoch)

            losses.update(loss.item(), inputs[0].size(0))
            # print(pids[0].size())
            # print(pids_eug.size())
            precisions.update(prec, pids[0].size(0) + pids_eug.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, w = inputs
        inputs = [imgs]
        if isinstance(pids, list):
            targets = [p.cuda() for p in pids]
        else:
            targets = pids.cuda()
        w = w.float().cuda()
        return inputs, targets, w

    def _forward(self, inputs, pids, inputs_eug, pids_eug, epoch, w_eug=None):
        loss_uns = 0.0
        loss_os = 0.0
        outputs = self.model(*inputs)
        loss_global, prec_global = self.criterions[1](outputs[1], pids[0], epoch)
        loss_uns += loss_global
        prec = prec_global

        if isinstance(outputs[0], list):
            for i, output_p in enumerate(outputs[0]):
                loss_tri, prec_tri = self.criterions[0](output_p, pids[i], epoch)
                loss_uns += loss_tri
        else:
            loss_tri, prec_tri = self.criterions[0](outputs[0], pids[0], epoch)
            loss_uns += loss_tri
        if len(outputs) == 3:
            if isinstance(outputs[2], list):
                for out in outputs[2]:
                    target_p = self.target_distribution(out)
                    loss_function = nn.KLDivLoss(size_average=False)
                    loss_uns += loss_function(out.log(), target_p) / out.shape[0]
                    
            else:
                target_p = self.target_distribution(outputs[2])
                loss_function = nn.KLDivLoss(size_average=False)
                loss_uns += loss_function(outputs[2].log(), target_p) / outputs[2].shape[0]
        
        outputs_eug = self.model(*inputs_eug)
        loss_global_eug, prec_global_eug = self.criterions[1](outputs_eug[1], pids_eug, epoch)
        loss_os += loss_global_eug
        prec += prec_global_eug

        if isinstance(outputs_eug[0], list):
            for i, output_p in enumerate(outputs_eug[0]):
                loss_tri, prec_tri = self.criterions[0](output_p, pids_eug, epoch)
                loss_os += loss_tri
            # loss_tri, prec_tri = self.criterions[0](outputs_eug[0][0], pids_eug, epoch)
            # loss_os += loss_tri
        else:
            loss_tri, prec_tri = self.criterions[0](outputs_eug[0], pids_eug, epoch)
            loss_os += loss_tri

        if len(outputs) == 3:
            if isinstance(outputs_eug[2], list):
                for out in outputs_eug[2]:
                    target_p = self.target_distribution(out)
                    loss_function = nn.KLDivLoss(size_average=False)
                    loss_os += loss_function(out.log(), target_p) / out.shape[0]
                    
            else:
                target_p = self.target_distribution(outputs_eug[2])
                loss_function = nn.KLDivLoss(size_average=False)
                loss_os += loss_function(outputs_eug[2].log(), target_p) / outputs_eug[2].shape[0]

        loss = loss_os + loss_uns
        return loss, prec

    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()