# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import directed_hausdorff

def re_ranking(input_feature_source, input_feature, k=20, lambda_value=0.1, MemorySave=False, Minibatch=2000):  ##rerank_plain
    all_num_source = input_feature_source.shape[0]
    all_num = input_feature.shape[0]    
    feat = input_feature.astype(np.float16)

    print('computing source distance...')
    sour_tar_dist = cdist(input_feature,input_feature_source)
    source_dist_vec = np.min(sour_tar_dist, axis = 1)
    source_dist_vec = source_dist_vec / np.max(source_dist_vec)
    source_dist = np.zeros([all_num, all_num])
    for i in range(all_num):
        source_dist[i,:] = source_dist_vec + source_dist_vec[i]
    del sour_tar_dist
    del source_dist_vec

    print('computing original distance...')
    if MemorySave:
        original_dist = np.zeros(shape=[all_num, all_num], dtype=np.float16)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it, ] = np.power(
                    cdist(feat[i:it, ], feat), 2).astype(np.float16)
            else:
                original_dist[i:, :] = np.power(
                    cdist(feat[i:, ], feat), 2).astype(np.float16)
                break
            i = it
    else:
        original_dist = cdist(feat, feat).astype(np.float16)
        original_dist = np.power(original_dist, 2).astype(np.float16)
    del feat

    euclidean_dist = original_dist/np.max(original_dist)
    ## compute k-nn of each instance
    knn_bool = np.zeros([all_num,all_num],dtype=bool)
    for i in range(all_num):
        tem_vec = original_dist[i,:]
        kThreshold = np.partition(tem_vec,k-1,)[k-1]
        knn_bool[i,:] = (tem_vec <= kThreshold)
        knn_bool[i,i] = False
        del tem_vec
    
    # new added for compute hausdorff distance matrix 
    hausdorff_dist = np.zeros([all_num,all_num], dtype=np.float64)
    for i in range(all_num):
        u_set = input_feature[knn_bool[i]]
        for j in range(i+1, all_num):
            v_set = input_feature[knn_bool[j]]                             
            hausdorff_dist[i,j] = max(directed_hausdorff(u_set, v_set)[0], directed_hausdorff(v_set, u_set)[0])
            hausdorff_dist[j,i] = hausdorff_dist[i,j]
        
    hausdorff_dist = hausdorff_dist/np.max(hausdorff_dist)

    final_dist = hausdorff_dist*(1-lambda_value) + source_dist*lambda_value
    del original_dist
    del hausdorff_dist
    return euclidean_dist, final_dist