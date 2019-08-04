import numpy as np
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def my_Allgather(comm, local_data, data=None):
    if not isinstance(local_data, np.ndarray):
        local_data = np.array(local_data)

    if data is None:
        data = np.empty(local_data.size * comm.Get_size(),
                        dtype=local_data.dtype)
    else:
        assert (data.size == local_data.size * comm.Get_size()
                and data.dtype == local_data.dtype)

    comm.Allgather(local_data, data)
    return data


def my_Allgatherv(local_data, comm=comm, data=None):
    if not isinstance(local_data, np.ndarray):
        local_data = np.array(local_data)

    sz_arr = my_Allgather(comm, local_data.size)
    os_arr = np.empty_like(sz_arr)
    os_arr[0] = 0
    os_arr[1:] = sz_arr[:-1].cumsum()

    if data is None:
        data = np.empty(sz_arr.sum(), dtype=local_data.dtype)
    else:
        assert (data.size == sz_arr.sum() and data.dtype == local_data.dtype)

    comm.Allgatherv(local_data, (data, (sz_arr, os_arr)))
    return data.reshape(-1, *local_data.shape[1:])


def LDOT(feat, proj):
    coord = np.empty((feat.shape[0], proj.shape[1]), dtype=proj.dtype)
    item_size = max(feat.dtype.itemsize, proj.itemsize)
    chunk_sz = int((1 << 30) / item_size / feat.shape[1])
    for si in range(0, feat.shape[0], chunk_sz):
        ei = min(feat.shape[0], si + chunk_sz)
        coord[si:ei] = np.dot(feat[si:ei], proj)
    return coord


def LEculidean(qry, ref):
    qry_sp = (qry ** 2).sum(axis=1).reshape(qry.shape[0], 1)
    ref_sp = (ref ** 2).sum(axis=1).reshape(1, ref.shape[0])
    dist = -2 * LDOT(qry, ref.T)
    dist += qry_sp
    dist += ref_sp
    return dist


def CalClassificationError_MPI(local_qry_feat, local_qry_label, ref_feat, ref_label, threshold_l, dist=None, fid=None):
    if dist is None:
        dist = LEculidean(local_qry_feat, ref_feat)
    else:
        assert dist.shape == (local_qry_feat.shape[0], ref_feat.shape[0])

    local_qry_num = local_qry_feat.shape[0]
    ref_num = ref_feat.shape[0]
    local_pos_err_num_arr = np.zeros(len(threshold_l))
    local_neg_err_num_arr = np.zeros(len(threshold_l))
    local_pos_num = local_neg_num = 0
    for lqi in range(local_qry_num):
        pos_indices = np.where(ref_label == local_qry_label[lqi])[0]
        neg_indices = np.where(ref_label != local_qry_label[lqi])[0]
        pos_dist = dist[lqi][pos_indices]
        neg_dist = dist[lqi][neg_indices]
        local_pos_num += pos_indices.size
        local_neg_num += neg_indices.size
        for idx, thres in enumerate(threshold_l):
            local_pos_err_num_arr[idx] += (pos_dist >= thres).sum()
            local_neg_err_num_arr[idx] += (neg_dist < thres).sum()
    pos_err_num_arr = comm.allreduce(local_pos_err_num_arr, op=MPI.SUM)
    neg_err_num_arr = comm.allreduce(local_neg_err_num_arr, op=MPI.SUM)
    pos_num = comm.allreduce(local_pos_num, op=MPI.SUM)
    neg_num = comm.allreduce(local_neg_num, op=MPI.SUM)

    pos_err_rate_arr = pos_err_num_arr.astype('float') / pos_num
    neg_err_rate_arr = neg_err_num_arr.astype('float') / neg_num
    log = 'pos pair num {}, neg pair num {}\n'.format(pos_num, neg_num)
    #for idx, thres in enumerate(threshold_l):
    #    log += 'Classification error with threshold {:.4f}: false_pos {:.4f}, false_neg {:.4f}, false_avg {:.4f}\n'.format(
    #        thres, pos_err_rate_arr[idx], neg_err_rate_arr[idx], 0.5 * (pos_err_rate_arr[idx] + neg_err_rate_arr[idx]))

    if comm_rank == 0:
        print(log)

    if fid is not None:
        fid.write(log + '\n')

    return pos_err_rate_arr, neg_err_rate_arr


def findMetricThreshold_MPI(local_qry_feat, local_qry_label, local_ref_feat, local_ref_label, dist=None, fid=None):
    ref_feat = my_Allgatherv(local_ref_feat)
    ref_label = my_Allgatherv(local_ref_label)
    if dist is None:
        dist = LEculidean(local_qry_feat, ref_feat)
    else:
        assert dist.shape == (local_qry_feat.shape[0], ref_feat.shape[0])
    dist[dist <= 0] = 0.0
    dist = np.sqrt(dist)

    local_qry_num = local_qry_feat.shape[0]
    ref_num = ref_feat.shape[0]

    local_intra_num = local_intra_sum = local_intra_sum2 = 0
    local_intra_min = 1E20
    local_intra_max = 0

    local_inter_num = local_inter_sum = local_inter_sum2 = 0
    local_inter_min = 1E20
    local_inter_max = 0

    local_intra_v = []
    local_inter_v = []
    for lqi in range(local_qry_num):
        intra_indices = np.where(ref_label == local_qry_label[lqi])[0]
        local_intra_dist = dist[lqi][intra_indices]
        local_intra_num += intra_indices.size
        local_intra_sum += local_intra_dist.sum()
        local_intra_sum2 += (local_intra_dist ** 2).sum()
        local_intra_min = min(local_intra_dist.min(), local_intra_min)
        local_intra_max = max(local_intra_dist.max(), local_intra_max)
        local_intra_v.append(local_intra_dist)

        inter_indices = np.where(ref_label != local_qry_label[lqi])[0]
        inter_dist = dist[lqi][inter_indices]
        local_inter_num += inter_indices.size
        local_inter_sum += inter_dist.sum()
        local_inter_sum2 += (inter_dist ** 2).sum()
        local_inter_min = min(inter_dist.min(), local_inter_min)
        local_inter_max = max(inter_dist.max(), local_inter_max)
        local_inter_v.append(inter_dist)

    qry_num = comm.allreduce(local_qry_num, op=MPI.SUM)
    intra_num = comm.allreduce(local_intra_num, op=MPI.SUM)
    intra_sum = comm.allreduce(local_intra_sum, op=MPI.SUM)
    intra_min = comm.allreduce(local_intra_min, op=MPI.MAX)
    intra_max = comm.allreduce(local_intra_max, op=MPI.MIN)
    intra_sum2 = comm.allreduce(local_intra_sum2, op=MPI.SUM)
    inter_num = comm.allreduce(local_inter_num, op=MPI.SUM)
    inter_sum = comm.allreduce(local_inter_sum, op=MPI.SUM)
    inter_min = comm.allreduce(local_inter_min, op=MPI.MIN)
    inter_max = comm.allreduce(local_inter_max, op=MPI.MAX)
    inter_sum2 = comm.allreduce(local_inter_sum2, op=MPI.SUM)
    intra_v = np.hstack(comm.allgather(local_intra_v))
    inter_v = np.hstack(comm.allgather(local_inter_v))

    intra_avg = intra_sum / intra_num
    intra_std = np.sqrt(intra_sum2 / intra_num - intra_avg ** 2)

    inter_avg = inter_sum / inter_num
    inter_std = np.sqrt(inter_sum2 / inter_num - inter_avg ** 2)

    qry_num = comm.allreduce(local_qry_num, op=MPI.SUM)
    if comm_rank == 0:
        print('Intra Distance: {}, {:.4f}+-{:.4f}, min {:.4f}, max {:.4f}'.format(intra_num, intra_avg, intra_std,
                                                                              intra_min, intra_max))
        print('Inter Distance: {}, {:.4f}+-{:.4f}, min {:.4f}, max {:.4f}'.format(inter_num, inter_avg, inter_std,
                                                                              inter_min, inter_max))

    if intra_avg >= inter_avg:
        print('The Metric Feature Is Too Bad!')
    else:
        thres = np.linspace(intra_avg, inter_avg, 10)
        CalClassificationError_MPI(
            local_qry_feat=local_qry_feat,
            local_qry_label=local_qry_label,
            ref_feat=ref_feat,
            ref_label=ref_label,
            threshold_l=thres,
            dist=dist)


    intra_v = np.hstack(intra_v)
    inter_v = np.sort(np.hstack(inter_v))
    FAR = [0.01, 0.001, 0.0001, 0.00001]
    if comm_rank == 0:
        # save FAR/GAR pairs
        if fid is not None:
            for num in range(len(inter_v)):
                thr = inter_v[num]
                cnt = len(intra_v[intra_v < thr])
                GAR = float(cnt) / intra_num
                fid.write("%.5f %.5f %.4f\n" % ((num + 1.0) / inter_num, GAR, thr))

        for k in range(len(FAR)):
            num = int(FAR[k] * inter_num)
            thr = inter_v[num]
            cnt = len(intra_v[intra_v < thr])
            GAR = float(cnt) / intra_num
            print("thr:%.4f  FAR:%.5f(%d/%d)  GAR:%.5f(%d/%d)" % (thr, FAR[k], num, inter_num, GAR, cnt, intra_num))

