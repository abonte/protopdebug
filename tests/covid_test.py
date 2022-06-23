import unittest

from data.cdatasets.bimcvcovid import BIMCVCOVIDDataset
import torch

from data.cdatasets.chestxray14h5 import ChestXray14H5Dataset
from data.cdatasets.githubcovid import GitHubCOVIDDataset
from data.cdatasets.padchesth5 import PadChestH5Dataset
from sklearn.utils import class_weight
import numpy as np


class CovidTest(unittest.TestCase):
    batch_size = 1000
    labels = 'chestx-ray14'
    fold = 'test'

    def test_chestxray14(self):
        c = ChestXray14H5Dataset(fold=self.fold, labels=self.labels, random_state=0,
                                 normalize=False, initialize_h5=True)

        c.init_worker(1)
        dataloader = torch.utils.data.DataLoader(
            c,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=c.init_worker)

        img, label, path, _ = next(iter(dataloader))
        print('ChestX-ray14', len(dataloader.dataset))
        assert len(dataloader.dataset) > 0
        assert len(label.shape) == 1
        assert torch.unique(label) == torch.tensor([0])  # only no-covid

    def test_githubcovid(self):
        c = GitHubCOVIDDataset(fold=self.fold, labels=self.labels, random_state=0,
                               normalize=False)
        c.init_worker(1)
        dataloader = torch.utils.data.DataLoader(
            c,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=c.init_worker)

        img, label, path, _ = next(iter(dataloader))
        print('GitHub-COVID', len(dataloader.dataset))
        assert len(dataloader.dataset) > 0
        assert len(label.shape) == 1
        assert torch.all(
            torch.unique(label).eq(torch.tensor([0, 1])))  # TODO FIX only covid

    def test_bimcv_covid(self):
        a = BIMCVCOVIDDataset(fold=self.fold, labels=self.labels, random_state=1)
        a.init_worker(1)
        dataloader = torch.utils.data.DataLoader(
            a,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=a.init_worker)

        img, label, _, _ = next(iter(dataloader))
        print('BIMCV+', len(dataloader.dataset))
        assert len(dataloader.dataset) > 0
        assert len(label.shape) == 1
        assert torch.unique(label) == torch.tensor([1])  # only covid

    def test_padchest(self):
        b = PadChestH5Dataset(fold=self.fold, labels=self.labels, random_state=0,
                              initialize_h5=True)
        b.init_worker(1)

        dataloader = torch.utils.data.DataLoader(
            b,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=b.init_worker)

        img, label, _, _ = next(iter(dataloader))
        print('PadChest', len(dataloader.dataset))
        assert len(dataloader.dataset) > 0
        assert len(label.shape) == 1
        assert torch.unique(label) == torch.tensor([0])  # only no covid

    def test_compute_class_weights(self):
        a = ChestXray14H5Dataset(fold='train', labels=self.labels, random_state=0,
                                 normalize=False, initialize_h5=True)

        b = GitHubCOVIDDataset(fold='train', labels=self.labels, random_state=0,
                               normalize=False)

        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes=[0, 1],
                                                          y=np.hstack([np.ones(len(b)),
                                                                       np.zeros(
                                                                           len(a))]))
        print(class_weights)


if __name__ == '__main__':
    unittest.main()
