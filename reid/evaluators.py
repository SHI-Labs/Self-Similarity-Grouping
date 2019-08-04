from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
# from .evaluation_metrics.eval_far_gar import findMetricThreshold_MPI
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_features(model, data_loader, print_freq=20, for_eval=True, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    end = time.time()
    for i, (imgs, fnames, pids, cams) in enumerate(data_loader):
        data_time.update(time.time() - end)
        outputs = extract_cnn_feature(model, imgs, for_eval)
        if (not for_eval ) and (isinstance(outputs, list)):
            outputs_ = extract_cnn_feature(model, fliplr(imgs), for_eval)
            for i in range(len(outputs)):
                out = outputs[i] + outputs_[i]
                fnorm = torch.norm(out, p=2, dim=1, keepdim=True)
                out = out.div(fnorm.expand_as(out))
                outputs[i] = out

            for index, (fname, pid, cam) in enumerate(zip(fnames, pids, cams)):
                features[fname] = [x[index] for x in outputs]
                labels[fname] = pid
        else:
            outputs += extract_cnn_feature(model, fliplr(imgs), for_eval)
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            outputs = outputs.div(fnorm.expand_as(outputs))
            for fname, output, pid, cam in zip(fnames, outputs, pids, cams):
                features[fname] = output
                labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    

    return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    #Compute all kinds of CMC scores
    cmc_configs = {
        # 'allshots': dict(separate_camera_set=False,
        #                  single_gallery_shot=False,
        #                  first_match_break=False),
        # 'cuhk03': dict(separate_camera_set=True,
        #                single_gallery_shot=True,
        #                first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    # print('CMC Scores{:>12}{:>12}{:>12}'
    #       .format('allshots', 'cuhk03', 'market1501'))
    # for k in cmc_topk:
    #     print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
    #           .format(k, cmc_scores['allshots'][k - 1],
    #                   cmc_scores['cuhk03'][k - 1],
    #                   cmc_scores['market1501'][k - 1]))
    print('CMC Scores{:>12}'.format('market1501'))
    for k in cmc_topk:
        print('top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))

    # Use the allshots cmc top-1 score for validation criterion
    # return cmc_scores['allshots'][0]
    return cmc_scores['market1501'][0]
    #return mAP


def evaluate_same_cams_all(distmat, query=None, gallery=None,
                           query_ids=None, gallery_ids=None,
                           cmc_topk=(1, 5, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [0 for _, _, _ in query]
        gallery_cams = [1 for _, _, _ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    #Compute all kinds of CMC scores
    cmc_configs = {
        # 'allshots': dict(separate_camera_set=False,
        #                  single_gallery_shot=False,
        #                  first_match_break=False),
        # 'cuhk03': dict(separate_camera_set=True,
        #                single_gallery_shot=True,
        #                first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    # print('CMC Scores{:>12}{:>12}{:>12}'
    #       .format('allshots', 'cuhk03', 'market1501'))
    # for k in cmc_topk:
    #     print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
    #           .format(k, cmc_scores['allshots'][k - 1],
    #                   cmc_scores['cuhk03'][k - 1],
    #                   cmc_scores['market1501'][k - 1]))
    print('CMC Scores{:>12}'.format('market1501'))
    for k in cmc_topk:
        print('top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))

    # Use the allshots cmc top-1 score for validation criterion
    # return cmc_scores['allshots'][0]
    return cmc_scores['market1501'][0]


class Evaluator(object):
    def __init__(self, model, print_freq):
        super(Evaluator, self).__init__()
        self.model = model
        self.print_freq = print_freq

    def evaluate(self, data_loader, query, gallery, metric=None):
        features, _ = extract_features(self.model, data_loader, print_freq=self.print_freq)
        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery)

    def evaluate_same_cams(self, data_loader, query, gallery, metric=None):
        features, _ = extract_features(self.model, data_loader)

        x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1).numpy()
        y = y.view(n, -1).numpy()
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        findMetricThreshold_MPI(x, query_ids, y, gallery_ids)

        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_same_cams_all(distmat, query=query, gallery=gallery)

