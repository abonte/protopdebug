import os
import unittest

import torch
from torchvision.transforms import transforms

import settings
from data.data_loader import ImageFolderWithFineAnnotation, ImageFolderWithPaths
from datasets import DATASETS


class DatasetTest(unittest.TestCase):
    img_size = 224
    img_path = os.path.join(settings.Cub200Clean5.data_path, settings.Cub200Clean5.train_directory)
    mask_path = img_path + '_segmentation'
    push_path = os.path.join(settings.Cub200Clean5.data_path, settings.Cub200Clean5.train_push_directory)
    generator = torch.Generator()
    generator.manual_seed(0)

    def test_ImageFolderWithFineAnnotation(self):
        transformation = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        dataset = ImageFolderWithFineAnnotation(img_path=self.img_path,
                                                mask_path=self.mask_path,
                                                annotated_rate=1,
                                                image_transform=transformation,
                                                mask_transform=transformation,
                                                rng=self.generator)

        batch_size = 100
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False)
        img, target, img_path, mask, with_fa = next(iter(loader))

        assert len(img) == batch_size
        assert len(target) == batch_size
        assert len(mask) == batch_size
        assert len(img_path) == batch_size
        assert len(with_fa) == batch_size
        assert img.shape[2] == mask.shape[1]
        assert img.shape[3] == mask.shape[2]

    def test_ImageFolderWithFineAnnotation_with_rate_0(self):
        transformation = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        dataset = ImageFolderWithFineAnnotation(img_path=self.img_path,
                                                mask_path=self.mask_path,
                                                push_path=self.push_path,
                                                annotated_rate=0,
                                                image_transform=transformation,
                                                mask_transform=transformation,
                                                rng=self.generator)

        batch_size = 100
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False)
        img, target, img_path, mask, with_fa = next(iter(loader))

        assert torch.max(mask) == 1 == torch.min(mask)  # no annotated images

    def test_ImageFolderWithPaths(self):
        transformation = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        dataset = ImageFolderWithPaths(self.img_path, transformation)

        batch_size = 100
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False)
        img, target, img_path = next(iter(loader))
        assert len(img) == batch_size
        assert len(target) == batch_size
        assert len(img_path) == batch_size

    def test_load_covid_dataset1(self):
        dataset = DATASETS['covid'](seed=0)
        inputs, labels, _, ds = next(iter(dataset.train_loader))
        assert inputs.shape[0] > 0
        assert inputs.shape[0] == labels.shape[0]


if __name__ == '__main__':
    unittest.main()
