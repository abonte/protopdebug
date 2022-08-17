import os
import random
from typing import Tuple, Any

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from sklearn.utils import Bunch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import IMG_EXTENSIONS
from sklearn.utils import class_weight

from data.cdatasets import ChestXray14H5Dataset
from data.cdatasets import DomainConfoundedDataset
from data.cdatasets.githubcovid import GitHubCOVIDDataset
from data.cdatasets.bimcvcovid import BIMCVCOVIDDataset
from data.cdatasets.padchesth5 import PadChestH5Dataset
from preprocess import mean, std
from settings import BaseDataset, Cub200ArtificialConfound, Cub200Clean5, \
    Cub200BackgroundConfound, SyntheticDatasetConfig, Cub200CleanAll, \
    Cub200CleanTop20, ExperimentConfig, CovidDatasetConfig


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DataModule:

    def __init__(self, cfg: ExperimentConfig, rng, global_analysis=False, device='cpu'):
        self.cfg_data: BaseDataset = cfg.data
        self.cfg = cfg
        self.global_analysis = global_analysis
        self.device = device
        self.num_worker = 0 if os.uname().nodename == 'mycomp.local' else 4
        self.rng = rng

        self.no_normalize_trans = transforms.Compose([
            transforms.Resize(size=(self.cfg_data.img_size, self.cfg_data.img_size)),
            transforms.ToTensor()
        ])

        self.normalize_trans = transforms.Compose([
            transforms.Resize(size=(self.cfg_data.img_size, self.cfg_data.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def prepare_data(self, loss_image=False):
        train_directory = os.path.join(self.cfg_data.data_path,
                                       self.cfg_data.train_directory)
        push_directory = os.path.join(self.cfg_data.data_path,
                                      self.cfg_data.train_push_directory)
        test_directory = os.path.join(self.cfg_data.data_path,
                                      self.cfg_data.test_directory)
        loss_directory = os.path.join(self.cfg_data.data_path,
                                      self.cfg_data.forbidden_protos_directory)
        remembering_loss_dir = os.path.join(self.cfg_data.data_path,
                                            self.cfg_data.remembering_protos_directory)

        if self.cfg.debug.loss == 'iaiabl':
            assert not self.cfg_data.train_on_segmented_image

        fa_rate = self.cfg.debug.fine_annotation if self.cfg.debug.loss == 'iaiabl' else 1

        self.train_data = ImageFolderWithFineAnnotation(img_path=train_directory,
                                                        mask_path=train_directory + '_segmentation',
                                                        push_path=push_directory,
                                                        annotated_rate=fa_rate,
                                                        image_transform=self.normalize_trans,
                                                        mask_transform=self.no_normalize_trans,
                                                        segment_image=self.cfg_data.train_on_segmented_image,
                                                        use_cache=True,
                                                        rng=self.rng,
                                                        device=self.device)

        self.val_data = None
        self.push_data = ImageFolderWithFineAnnotation(img_path=push_directory,
                                                       mask_path=push_directory + '_segmentation',
                                                       image_transform=self.no_normalize_trans,
                                                       mask_transform=self.no_normalize_trans,
                                                       segment_image=False,
                                                       use_cache=False,
                                                       device=self.device)

        self.loss_data = None
        self.positive_loss_data = None
        if loss_image:
            self.loss_data = ImageFolderLoss(loss_directory, self.normalize_trans,
                                             self.device)
            if os.path.exists(remembering_loss_dir):
                self.positive_loss_data = ImageFolderLoss(remembering_loss_dir,
                                                          self.normalize_trans,
                                                          self.device)

        self.kernel_set_data = None
        if self.global_analysis:
            # must use unaugmented (original) dataset
            # train set: do not normalize
            # test set: do not normalize
            self.test_data = ImageFolderWithFineAnnotation(img_path=test_directory,
                                                           mask_path=test_directory + '_segmentation',
                                                           image_transform=self.no_normalize_trans,
                                                           mask_transform=self.no_normalize_trans,
                                                           segment_image=False,
                                                           use_cache=False,
                                                           device=self.device)
            if os.path.exists(os.path.join(self.cfg_data.data_path, 'kernel_set')):
                self.kernel_set_data = datasets.ImageFolder(
                    os.path.join(self.cfg_data.data_path, 'kernel_set'),
                    self.no_normalize_trans)
        elif self.cfg_data.test_on_segmented_image:
            self.test_data = ImageFolderWithFineAnnotation(img_path=test_directory,
                                                           mask_path=test_directory + '_segmentation',
                                                           image_transform=self.normalize_trans,
                                                           mask_transform=self.no_normalize_trans,
                                                           segment_image=True,
                                                           device=self.device)
        else:
            self.test_data = datasets.ImageFolder(test_directory, self.normalize_trans)

    def train_dataloader(self):
        def set_seed(seed=0):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        set_seed(0)
        return DataLoader(self.train_data,
                          num_workers=0,
                          worker_init_fn=seed_worker,
                          batch_size=self.cfg_data.train_batch_size,
                          shuffle=True,
                          generator=self.rng)

    def val_dataloader(self):
        pass

    def test_datalaoder(self):
        return DataLoader(
            self.test_data,
            batch_size=self.cfg_data.test_batch_size,
            shuffle=True,  # True because of global anaysis by in main is False, fix?
            num_workers=self.num_worker,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=self.rng)

    def push_dataloader(self):
        return DataLoader(
            self.push_data,
            batch_size=self.cfg_data.train_push_batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=self.rng)

    def kernelset_dataloader(self):
        if self.kernel_set_data is not None:
            return DataLoader(self.kernel_set_data, batch_size=1000)
        else:
            return None

    def loss_dataloader(self):
        if self.loss_data is not None:
            return self.loss_data.get_all()
        else:
            return None

    def positive_loss_dataloader(self):
        if self.positive_loss_data is not None:
            return self.positive_loss_data.get_all()
        else:
            return None

    def get_dataset(self):
        return Bunch(train_loader=self.train_dataloader(),
                     train_push_loader=self.push_dataloader(),
                     test_loader=self.test_datalaoder(),
                     img_size=self.cfg_data.img_size,
                     loss_loader=self.loss_dataloader(),
                     positive_loss_loader=self.positive_loss_dataloader(),
                     num_classes=self.cfg_data.num_classes,
                     kernel_set_loader=self.kernelset_dataloader(),
                     class_weights=None)


class CovidDataModule:

    def __init__(self, cfg: ExperimentConfig, generator, global_analysis, device='cpu'):
        self.cfg_data: BaseDataset = cfg.data
        self.seed = cfg.seed
        self.global_analysis = global_analysis
        self.device = device

    def prepare_data(self, loss_image=False):
        labels = 'chestx-ray14'
        if self.global_analysis:
            trainpushds = DomainConfoundedDataset(
                ChestXray14H5Dataset(fold='train', labels=labels,
                                     random_state=self.seed, normalize=False,
                                     initialize_h5=True),
                GitHubCOVIDDataset(fold='train', labels=labels, random_state=self.seed,
                                   normalize=False)
            )

            self.train_push_loader = torch.utils.data.DataLoader(
                trainpushds,
                batch_size=self.cfg_data.train_push_batch_size,
                num_workers=2,
                worker_init_fn=trainpushds.init_worker)

            bimcvpadchest_testds = DomainConfoundedDataset(
                PadChestH5Dataset(fold='train', labels='chestx-ray14',
                                  random_state=self.seed, initialize_h5=True),
                BIMCVCOVIDDataset(fold='train', labels='chestx-ray14',
                                  random_state=self.seed, initialize_h5=True)
            )

            self.test_dataloader = torch.utils.data.DataLoader(
                bimcvpadchest_testds,
                batch_size=self.cfg_data.test_batch_size,
                shuffle=False,
                num_workers=8,
                worker_init_fn=bimcvpadchest_testds.init_worker)

            if loss_image:
                loss_directory = os.path.join(self.cfg_data.data_path,
                                              self.cfg_data.forbidden_protos_directory)
                self.loss_data = ImageFolderLoss(loss_directory,
                                                 trainpushds.ds1._transforms['push'],
                                                 self.device)
        else:
            trainds = DomainConfoundedDataset(
                ChestXray14H5Dataset(fold='train', labels=labels,
                                     random_state=self.seed, initialize_h5=True),
                GitHubCOVIDDataset(fold='train', labels=labels, random_state=self.seed)
            )

            covid_negative = ChestXray14H5Dataset(fold='train', labels=labels,
                                                  random_state=self.seed,
                                                  normalize=False, initialize_h5=True)

            covid_positive = GitHubCOVIDDataset(fold='train', labels=labels,
                                                random_state=0,
                                                normalize=False)

            self.class_weights = torch.from_numpy(class_weight.compute_class_weight(
                'balanced',
                classes=[0, 1],
                y=np.hstack(
                    [np.ones(len(covid_positive)), np.zeros(len(covid_negative))]))) \
                .to(self.device)

            trainpushds = DomainConfoundedDataset(
                ChestXray14H5Dataset(fold='train', labels=labels,
                                     random_state=self.seed,
                                     normalize=False, initialize_h5=True),
                GitHubCOVIDDataset(fold='train', labels=labels, random_state=self.seed,
                                   normalize=False)
            )

            valds = DomainConfoundedDataset(
                ChestXray14H5Dataset(fold='val', labels=labels, random_state=self.seed,
                                     initialize_h5=True),
                GitHubCOVIDDataset(fold='val', labels=labels, random_state=self.seed)
            )

            bimcvpadchest_testds = DomainConfoundedDataset(
                PadChestH5Dataset(fold='train', labels='chestx-ray14',
                                  random_state=self.seed, initialize_h5=True),
                BIMCVCOVIDDataset(fold='train', labels='chestx-ray14',
                                  random_state=self.seed, initialize_h5=True)
            )

            self.train_dataloader = torch.utils.data.DataLoader(
                trainds,
                batch_size=self.cfg_data.train_batch_size,
                shuffle=True,
                num_workers=2,
                worker_init_fn=trainds.init_worker)

            self.train_push_loader = torch.utils.data.DataLoader(
                trainpushds,
                batch_size=self.cfg_data.train_push_batch_size,
                num_workers=2,
                worker_init_fn=trainpushds.init_worker)

            self.val_dataloader = torch.utils.data.DataLoader(
                valds,
                batch_size=self.cfg_data.test_batch_size,
                shuffle=False,
                num_workers=2,
                worker_init_fn=valds.init_worker)

            self.test_dataloader = torch.utils.data.DataLoader(
                bimcvpadchest_testds,
                batch_size=self.cfg_data.test_batch_size,
                shuffle=False,
                num_workers=8,
                worker_init_fn=bimcvpadchest_testds.init_worker)

            self.loss_data = None
            self.positive_loss_data = None

            if loss_image:
                ds1 = ChestXray14H5Dataset(fold='train', labels=labels,
                                           random_state=self.seed, initialize_h5=True)

                loss_directory = os.path.join(self.cfg_data.data_path,
                                              self.cfg_data.forbidden_protos_directory)
                self.loss_data = ImageFolderLoss(loss_directory,
                                                 ds1._transforms['train'],
                                                 self.device)

                remembering_loss_dir = os.path.join(self.cfg_data.data_path,
                                                    self.cfg_data.remembering_protos_directory)
                if os.path.exists(remembering_loss_dir):
                    self.positive_loss_data = ImageFolderLoss(remembering_loss_dir,
                                                              ds1._transforms['train'],
                                                              self.device)

    def get_dataset(self):
        if self.global_analysis:
            return Bunch(train_push_loader=self.train_push_loader,
                         test_loader=self.test_dataloader,
                         loss_loader=torch.utils.data.DataLoader(self.loss_data,
                                                                 batch_size=30))
        else:
            return Bunch(train_loader=self.train_dataloader,
                         train_push_loader=self.train_push_loader,
                         test_loader=self.test_dataloader,
                         val_loader=self.val_dataloader,
                         num_classes=self.cfg_data.num_classes,
                         img_size=self.cfg_data.img_size,
                         loss_loader=self.loss_data.get_all() if self.loss_data is not None else None,
                         positive_loss_loader=self.positive_loss_data.get_all() if self.positive_loss_data is not None else None,
                         class_weights=self.class_weights)


def get_data(cfg, generator, global_analysis, device='cpu') -> DataModule:
    return DataModule(cfg, generator, global_analysis, device)


def get_covid(cfg: ExperimentConfig, generator, global_analysis, device='cpu'):
    return CovidDataModule(cfg, generator, global_analysis, device)


DATASETS = {
    SyntheticDatasetConfig.name: get_data,
    Cub200ArtificialConfound.name: get_data,
    Cub200Clean5.name: get_data,
    Cub200CleanTop20.name: get_data,
    Cub200CleanAll.name: get_data,
    Cub200BackgroundConfound.name: get_data,
    CovidDatasetConfig.name: get_covid
}


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        sample, target = super(ImageFolderWithPaths, self).__getitem__(index)
        img_path = self.imgs[index][0]
        return sample, target, img_path


class ImageFolderLoss(datasets.ImageFolder):
    """Dataset used for forgetting loss"""

    def __init__(self, loss_directory, img_transformation, device):
        super(ImageFolderLoss, self).__init__(loss_directory, img_transformation)
        self.pre_loaded_imgs = []
        self.pre_loaded_targets = []
        for index in range(len(self.samples)):
            sample, target = super(ImageFolderLoss, self).__getitem__(index)
            self.pre_loaded_imgs.append(sample)
            self.pre_loaded_targets.append(target)

        self.pre_loaded_imgs = torch.stack(self.pre_loaded_imgs).to(device)

    def find_classes(self, directory):
        classes, class_to_idx = super(ImageFolderLoss, self).find_classes(directory)
        class_to_idx = {cls_name: int(cls_name.split('class_idx_')[-1]) for cls_name in
                        classes}
        return classes, class_to_idx

    def get_all(self):
        return self.pre_loaded_imgs, self.pre_loaded_targets


class ImageFolderWithFineAnnotation(datasets.ImageFolder):
    """For training datasets used by IAIA-BL
    fine annotation_image: white (1) relevant region, black (0) irrelevant region
    mask mi in iaiabl loss: 0 -> relevant region, 1 irrelevant region

    segment image: images with only bird and no background, used for experiment
    with natural confounds (e.g., sky)
    """

    def __init__(self, img_path: str,
                 mask_path: str,
                 image_transform,
                 mask_transform,
                 push_path: str = None,
                 annotated_rate: float = 1,
                 segment_image: bool = False,
                 use_cache: bool = False,
                 rng=None,
                 device='cpu'):

        self.use_cache = use_cache
        self.device = device
        super(ImageFolderWithFineAnnotation, self).__init__(img_path,
                                                            image_transform)
        self.img_path = img_path
        self.mask_path = mask_path
        self.mask_transformation = mask_transform
        self.segment_image = segment_image

        if push_path is not None:
            # collect idx images with fine annotation
            _, classes_with_masks_to_idx = self.find_classes(mask_path)
            images_classes, _ = self.find_classes(img_path)
            images = self.make_dataset(img_path, classes_with_masks_to_idx,
                                       IMG_EXTENSIONS)
            if annotated_rate > 1:
                n_selected_images = int(annotated_rate)
                assert n_selected_images <= len(images)
            else:
                n_selected_images = int(annotated_rate * len(images))
            idx_imgs = torch.randperm(len(images), generator=rng)[
                       :n_selected_images]

            annotate_images = []
            for idx, (path, _) in enumerate(images):
                if idx in idx_imgs:
                    annotate_images.append(path)

            self.imgs_with_masks = {}
            for idx, (path, cls) in enumerate(self.imgs):
                self.imgs_with_masks[idx] = None
                image_name = os.path.basename(path)
                if any([os.path.basename(p) == image_name for p in annotate_images]):
                    self.imgs_with_masks[idx] = os.path.join(self.mask_path,
                                                             self.classes[cls],
                                                             image_name)

        else:
            assert annotated_rate == 1
            classes_with_masks, _ = self.find_classes(mask_path)
            self.imgs_with_masks = {}
            for idx, (path, cls) in enumerate(self.imgs):
                self.imgs_with_masks[idx] = None
                if self.classes[cls] in classes_with_masks:
                    self.imgs_with_masks[idx] = os.path.join(self.mask_path,
                                                             self.classes[cls],
                                                             os.path.basename(path))

        assert not annotated_rate < 0

        if segment_image:
            if annotated_rate != 1:
                raise ValueError("fine annotation rate != 1 with segment image True")
            # all imgs need to be annotated
            assert sum([v is not None for _, v in self.imgs_with_masks.items()]) == len(
                self.imgs)

        self.cached_data = {}

    def __getitem__(self, index) -> Tuple[Any, Any, str, Any, bool]:
        if self.use_cache:
            try:
                return self.cached_data[index]
            except KeyError:
                segmented_sample, target, img_path, mask, with_fine_annotation = self._load_item(
                    index)
                sample = segmented_sample.to(
                    self.device), target, img_path, mask, with_fine_annotation
                self.cached_data[index] = sample
                return sample
        else:
            return self._load_item(index)

    def _load_item(self, index) -> Tuple[Any, Any, str, Any, bool]:
        sample, target = super(ImageFolderWithFineAnnotation, self).__getitem__(index)
        img_path = self.imgs[index][0]
        if self.imgs_with_masks[index] is not None:
            # image with a mask
            mask_path = self.imgs_with_masks[index]
            mask = self.mask_transformation(
                Image.open(mask_path).convert('L')).squeeze()

            # handle mask without white
            clip_threshold = np.median(np.unique(mask)) if 1 not in mask else 0.7

            mask[mask <= clip_threshold] = 0
            mask[mask > clip_threshold] = 1

            with_fine_annotation = True

            assert os.path.splitext(os.path.basename(mask_path))[0] == \
                   os.path.splitext(os.path.basename(self.imgs[index][0]))[0]
            assert np.all((np.unique(mask) == 0) | (np.unique(mask) == 1))
        else:
            # no mask available
            mask = torch.ones((sample.shape[1], sample.shape[2]))
            with_fine_annotation = False

        if mask.shape != sample.shape[1:]:
            raise ValueError('Mask and image have different shape')

        if self.segment_image:
            segmented_sample = sample * mask
        else:
            segmented_sample = sample

        # debugging
        # transforms.ToPILImage(mode='RGB')(segmented_sample).show()
        # transforms.ToPILImage(mode='L')(mask).show()

        return segmented_sample, target, img_path, mask, with_fine_annotation


class ImageData(datasets.ImageFolder):

    def __getitem__(self, index) -> Tuple[Any, Any, str, Any, bool]:
        sample, target = super(ImageData, self).__getitem__(index)
        return sample, target, str(index), torch.ones(
            (sample.shape[1], sample.shape[2])), True


def get_data(cfg, generator, global_analysis, device='cpu') -> DataModule:
    return DataModule(cfg, generator, global_analysis, device)
