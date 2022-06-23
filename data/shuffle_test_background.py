#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image
from omegaconf.omegaconf import OmegaConf
from torchvision.transforms import transforms

import settings
from settings import PATH_TO_RAW_CUB_200
from helpers import makedir


def extract_background_foreground(original_test_folder, cls_folder, img_name, resize):
    img_path = os.path.join(PATH_TO_RAW_CUB_200, 'images', cls_folder, img_name)
    mask_path = os.path.join('segmentations', cls_folder, img_name.replace('jpg', 'png'))

    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    mask = np.asarray(mask, dtype='uint8')
    mask = mask / 255

    # mask without white
    clip_threshold = np.median(np.unique(mask)) if 1 not in mask else 0.7
    mask[mask < clip_threshold] = 0
    mask[mask >= clip_threshold] = 1
    mask = np.asarray(mask, dtype='uint8')
    assert np.all((np.unique(mask) == 0) | (np.unique(mask) == 1))

    img = np.asarray(resize(img))
    resize2 = transforms.Compose(
        [transforms.ToPILImage(mode='L'),
         transforms.Resize(size=(cfg.img_size, cfg.img_size))])
    mask_bg_original = np.asarray(resize2(mask), dtype=int)

    bg = img * (1 - mask_bg_original[..., None])
    imarray = np.random.rand(cfg.img_size, cfg.img_size, 3) * 255

    bg = bg + (imarray * mask_bg_original[..., None])

    # =================================================

    img_path = os.path.join(original_test_folder, cls_folder, img_name)
    mask_path = os.path.join(original_test_folder + '_segmentation', cls_folder,
                             img_name)

    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    mask = np.asarray(mask, dtype='uint8')
    mask = mask / 255

    # mask without white
    clip_threshold = np.median(np.unique(mask)) if 1 not in mask else 0.7
    mask[mask < clip_threshold] = 0
    mask[mask >= clip_threshold] = 1
    mask = np.asarray(mask, dtype='uint8')
    assert np.all((np.unique(mask) == 0) | (np.unique(mask) == 1))

    img = np.asarray(resize(img))
    resize2 = transforms.Compose(
        [transforms.ToPILImage(mode='L'),
         transforms.Resize(size=(cfg.img_size, cfg.img_size))])
    mask_bg = np.asarray(resize2(mask), dtype=int)

    fg = img * mask_bg[..., None]

    return bg.astype('uint8'), fg.astype('uint8'), mask_bg.astype('uint8')


def paste_images(bg, fg, image_save_path, mask_save_path):
    _, fg, mask_fg = fg
    bg, _, _ = bg
    tmp2 = bg * (1 - mask_fg[..., None]) + fg
    Image.fromarray(tmp2.astype('uint8'), mode='RGB').save(image_save_path)
    Image.fromarray(mask_fg.astype('uint8') * 255).convert('1').save(mask_save_path)


def shuffle_bg(cfg: settings.Cub200CleanAll):
    # using torch to be sure that the same preprocessing is used (see torch.resize docs)
    resize = transforms.Compose(
        [transforms.Resize(size=(cfg.img_size, cfg.img_size))])

    original_test_folder = os.path.join(cfg.data_path, cfg.test_directory)
    dest = os.path.join(cfg.data_path, 'test_cropped_shuffled')
    dest_seg = dest + '_segmentation'

    cls_fg = os.listdir(original_test_folder)
    cls_bg = os.listdir(original_test_folder)
    cls_bg.append(cls_bg.pop(0))
    for cls_fg, cls_bg in zip(cls_fg, cls_bg):
        print(cls_bg, '-->', cls_fg)
        makedir(os.path.join(dest, cls_fg))
        makedir(os.path.join(dest_seg, cls_fg))
        imgs_name_bg = os.listdir(os.path.join(original_test_folder, cls_bg))
        for i, image_name_fg in enumerate(
                os.listdir(os.path.join(original_test_folder, cls_fg))):
            image_name_bg = imgs_name_bg[min(i, len(imgs_name_bg) - 1)]
            fg = extract_background_foreground(original_test_folder, cls_fg,
                                               image_name_fg, resize)
            bg = extract_background_foreground(original_test_folder, cls_bg,
                                               image_name_bg, resize)

            image_save_path = os.path.join(dest, cls_fg, image_name_fg)
            mask_save_path = os.path.join(dest_seg, cls_fg, image_name_fg)
            paste_images(bg, fg, image_save_path, mask_save_path)


if __name__ == '__main__':
    cfg: settings.Cub200CleanAll = OmegaConf.structured(settings.Cub200CleanAll)
    shuffle_bg(cfg)
