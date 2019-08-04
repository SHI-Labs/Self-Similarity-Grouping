from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import read_json
from ..utils.serialization import write_json


def _pluck(identities, indices, relabel=False):
    """Extract im names of given pids.
    Args:
      identities: containing im names
      indices: pids
      relabel: whether to transform pids to classification labels
    """
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                else:
                    ret.append((fname, pid, camid))
    return ret


class Belle(Dataset):

    def __init__(self, root, split_id=0, num_val=100, download=False):
        super(Belle, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        # Extract the file
        exdir = osp.join(self.root, 'raw')
        # exdir = '/data/liangchen.song/reid/belle'

        # Format
        images_dir = osp.join(self.root, 'images')
        # images_dir = '/data/liangchen.song/reid/belle/images'

        identities = [[[] for _ in range(1)] for _ in range(17164)]

        def register(subdir, pattern=re.compile(r'([\d]+)_')):
            fnames = []  # Added. Names of images in new dir.
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid = int(pattern.search(fname).groups()[0])
                assert 1 <= pid <= 17163  # pid == 0 means background
                cam = 0
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
                fnames.append(fname)  # Added
            return pids, fnames

        trainval_pids, _ = register('train')
        gallery_pids, gallery_fnames = register('gallery')
        query_pids, query_fnames = register('query')
        assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'belle', 'shot': 'multiple', 'num_cameras': 1,
                'identities': identities,
                'query_fnames': query_fnames,  # Added
                'gallery_fnames': gallery_fnames}  # Added
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))

    ########################
    # Added
    def load(self, verbose=True):
        import numpy as np
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split trainsubset
        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        
        subset_num = 684
        train_subset_pids = [[] for i in range(24)]
        for i in range(23):
            train_subset_pids[i] = sorted(trainval_pids[i*subset_num:(i+1)*subset_num])
        train_subset_pids[23] = sorted(trainval_pids[23*subset_num:])

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']

        self.train_subset = [[] for i in range(24)]
        self.train_subset_ids = [[] for i in range(24)]
        for i in range(24):
            self.train_subset[i] = _pluck(identities, train_subset_pids[i], relabel=True)
            self.train_subset_ids[i] = len(train_subset_pids[i])
        self.trainval = _pluck(identities, trainval_pids, relabel=True)
        self.num_trainval_ids = len(trainval_pids)

        ##########
        # Added
        query_fnames = self.meta['query_fnames']
        gallery_fnames = self.meta['gallery_fnames']
        self.query = []
        for fname in query_fnames:
            name = osp.splitext(fname)[0]
            pid, cam, _ = map(int, name.split('_'))
            self.query.append((fname, pid, cam))
        self.gallery = []
        for fname in gallery_fnames:
            name = osp.splitext(fname)[0]
            pid, cam, _ = map(int, name.split('_'))
            self.gallery.append((fname, pid, cam))
        ##########

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            for i in range(24):
                print("  train_{:2d} | {:5d} | {:8d}"
                       .format(i, self.train_subset_ids[i], len(self.train_subset[i])))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))
    ########################
