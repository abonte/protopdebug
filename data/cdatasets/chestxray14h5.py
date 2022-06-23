#!/usr/bin/env python
import io
import os

import h5py
import numpy
from PIL import Image

from settings import ROOT_DIR
from .chestxray14dataset import ChestXray14Dataset
from .cxrdataset import H5Dataset


class ChestXray14H5Dataset(ChestXray14Dataset, H5Dataset):
    """
    HDF5 dataset for ChestX-ray14 images.
    """

    def __init__(
            self,
            fold,
            random_state=30493,
            labels='CheXpert',
            initialize_h5=False,
            pneumo=None,
            normalize=False,
            subsample_test=True):
        """
        initialize_h5: (bool) If true, open a handle for the HDF5 file when the
            class is instantiated. Use `False` when the dataset will be wrapped
            by a PyTorch dataloader with num_workers > 0, and True otherwise.
        """
        self.h5path = os.path.join(ROOT_DIR, "datasets/covid/ChestX-ray14/chestxray14.h5")
        if initialize_h5:
            self.h5 = h5py.File(self.h5path, 'r', swmr=True)
        super().__init__(fold,
                         random_state=random_state,
                         labels=labels,
                         pneumo=pneumo,
                         normalize=normalize,
                         subsample_test=subsample_test)

    def init_worker(self, worker_id):
        self.h5 = h5py.File(self.h5path, 'r', swmr=True)

    def _raw_image_from_disk(self, idx):
        """
        Retrieve the raw PIL image from storage.
        """
        imagename = self.df.index[idx]
        data = self.h5['images']['images'].get(imagename)
        image = Image.open(io.BytesIO(numpy.array(data)))
        image = image.convert('RGB')
        return image, os.path.join(ROOT_DIR, "datasets/covid/ChestX-ray14/images/", imagename)
