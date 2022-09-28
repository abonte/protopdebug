import argparse
import glob
import os
import re
import shutil
import random

import cv2
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from omegaconf.omegaconf import OmegaConf

import settings
from automatic_extract_confound import extract_highest_activated_patches
from helpers import makedir, load


def prepare_cut_out_image(path_to_image, class_idx, dest, random_bg=False):
    size_in_original_img_of_one_pixel_in_latent_space = (32, 32)
    img = Image.open(os.path.join(path_to_image))

    # TODO fix to right receptive field
    # img = T.CenterCrop(size=size_in_original_img_of_one_pixel_in_latent_space)(img)
    img_w, img_h = img.size
    if random_bg:
        imarray = np.random.rand(cfg.img_size, cfg.img_size, 3) * 255
        background = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    else:
        background = Image.new(mode="RGB",
                               size=(cfg.img_size, cfg.img_size),
                               color='white')
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)

    _save_image_to_forbidden_or_remembering_folder(cfg, class_idx, background, dest)


def _save_image_to_forbidden_or_remembering_folder(cfg, cls_idx: int, image: Image, dest: str,
                                                   save_name: str = None):
    if dest == 'confound':
        dst_path = pjoin(cfg.data_path, cfg.forbidden_protos_directory,
                         f'class_idx_{cls_idx}')
    elif dest == 'remember':
        dst_path = pjoin(cfg.data_path, cfg.remembering_protos_directory,
                         f'class_idx_{cls_idx}')
    else:
        raise ValueError(dest)

    makedir(dst_path)
    if save_name is None:
        save_name = f'{len(os.listdir(dst_path))}out.png'
    image.save(os.path.join(dst_path, save_name))

    # dst_path = os.path.join(base_path+'_original', f'class_idx_{cls_idx}')
    # makedir(dst_path)
    # dr = os.path.split(args.path_to_image)[0]
    # shutil.copy(src=os.path.join(dr, f'prototype-img-original_with_self_act{prototype_idx}.png'),
    #             dst=os.path.join(dst_path, f'{len(os.listdir(dst_path))}prototype-img-original_with_self_act{prototype_idx}.png'))


def patches_extraction(path: str,
                       selected_classes_idx: list,
                       n_images_to_show: int = 5,
                       apr_threshold: float = 0.5):
    # selected_classes_idx = [8, 4, 5, 11, 1]
    # selected_classes_idx = [0,8,14,6,15]
    user_exp_path = os.path.join(path, 'user_experiment')
    makedir(os.path.join(user_exp_path, 'form_figures'))
    makedir(os.path.join(user_exp_path, 'cuts'))

    if selected_classes_idx is not None:
        print(f'Warning: show only {selected_classes_idx}')
    img_class_idxs = np.load(os.path.join(path, "full_class_id.npy"))
    print(img_class_idxs.shape)
    proto_class_identity = \
        load(os.path.join(os.path.dirname(path), 'stat.pickle'))['proto_identity']

    n_prototypes, _ = img_class_idxs.shape

    max_k = min(img_class_idxs.shape[1], n_images_to_show)

    for proto_idx, row in zip(range(0, n_prototypes), img_class_idxs):
        proto_class_idx = np.where(proto_class_identity[proto_idx] == 1)[0][0]
        if selected_classes_idx is not None:
            if proto_class_idx not in selected_classes_idx:
                continue

        dest_path = os.path.join(path, str(proto_idx), 'auto_patch_extraction')
        makedir(dest_path)
        for k in range(1, max_k + 1):
            print('================================')
            print(f'cl={proto_class_idx} pr={proto_idx} i={k}')

            act_filename = pjoin(path, str(proto_idx), f"nearest-{k}_act.pickle")
            if not os.path.exists(act_filename):
                continue
            act, _ = load(
                pjoin(path, str(proto_idx), f"nearest-{k}_act.pickle"))

            original_img = cv2.imread(
                os.path.join(path, str(proto_idx), f'nearest-{k}_original.png'))
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

            upsampled_act_img = cv2.resize(act,
                                           dsize=(original_img.shape[0],
                                                  original_img.shape[0]),
                                           interpolation=cv2.INTER_CUBIC)

            file_basename = f'c={proto_class_idx}_p={proto_idx}_i={k}'
            extract_highest_activated_patches(original_img,
                                              original_img.shape[0],
                                              upsampled_act_img,
                                              file_basename,
                                              dest_path,
                                              k)

        for exp_image in glob.glob(os.path.join(dest_path, f'exp_*')):
            shutil.copy(src=exp_image,
                        dst=os.path.join(user_exp_path, 'form_figures'))
        for exp_image in glob.glob(os.path.join(dest_path, f'c=*')):
            shutil.copy(src=exp_image,
                        dst=os.path.join(user_exp_path, 'cuts'))

    # randomize figure order for the experiment with real users
    makedir(os.path.join(user_exp_path, 'random_form_figures'))
    img_list = os.listdir(os.path.join(user_exp_path, 'form_figures'))
    img_list.sort()
    img_list = [name for name in img_list if name.startswith('exp_')]
    random_img_list = random.sample(img_list, len(img_list))
    for i, img in enumerate(random_img_list):
        shutil.copy(src=os.path.join(user_exp_path, 'form_figures', img),
                    dst=os.path.join(user_exp_path, 'random_form_figures',
                                     f'{str(i).zfill(2)}H{img}'))


def interactive_debugging(path):
    conf_dest_folder = pjoin(path, '..', 'tmp_forbidden_conf')
    remember_dest_folder = pjoin(path, '..', 'tmp_remember_patch')
    makedir(conf_dest_folder)
    makedir(remember_dest_folder)

    user_exp_path = os.path.join(path, 'user_experiment')
    img_paths = os.listdir(os.path.join(user_exp_path, 'form_figures'))
    img_paths.sort()

    n_confounded_patch, n_no_confounded_path, n_remember_patch = 0, 0, 0
    for img_name in img_paths:
        class_idx, proto_idx, img_idx = re.findall("\d+", img_name)[:3]
        act, act_pr = load(
            pjoin(path, str(proto_idx), f"nearest-{img_idx}_act.pickle"))
        fig, axs = plt.subplots()
        axs.set_title(
            f'img_cls={class_idx} p={proto_idx} i={img_idx}\n'
            f'max_act={np.max(act):.2f}\n'
            f'mean_act={np.mean(act):.2f}\n'
            f'apr={act_pr if act_pr is None else round(act_pr, 2)}')
        axs.imshow(plt.imread(os.path.join(user_exp_path, 'form_figures', img_name)))
        axs.axis('off')

        plt.show(block=False)

        path_to_img = os.path.join(user_exp_path, 'cuts',
                                   img_name.split('exp_')[1])
        #print(img_name.split('exp_')[1])
        while True:
            select = input(
                f'is this patch from image c={class_idx} i={img_idx} confounded? '
                f'[y (confounded) |n (no, go next) |r (remember)] ')
            if select == 'y':
                n_confounded_patch += 1
                shutil.copy(src=path_to_img, dst=conf_dest_folder)
                break
            elif select == 'n':
                n_no_confounded_path += 1
                break
            elif select == 'r':
                n_remember_patch += 1
                shutil.copy(src=path_to_img, dst=remember_dest_folder)
                break
            else:
                print(f'Wrong value {select}')
                continue
        plt.close(fig)
    print(f'\nStats: confounded={n_confounded_patch} '
          f'no={n_no_confounded_path} '
          f'remember={n_remember_patch}')


def move_patches_to_forbidden_remember_folder(path):
    conf_dest_folder = pjoin(path, 'tmp_forbidden_conf')
    remember_dest_folder = pjoin(path, 'tmp_remember_patch')
    for source_folder, dest_folder, type in [
        (conf_dest_folder, cfg.forbidden_protos_directory, 'confound'),
        (remember_dest_folder, cfg.remembering_protos_directory, 'remember')
    ]:
        if not os.path.exists(source_folder):
            continue
        for img_name in os.listdir(source_folder):
            if img_name in ['.DS_Store'] or 'act' in img_name:
                print(f'skip {img_name}')
                continue
            class_idx, _, _ = re.findall("\d+", img_name)[:3]
            if 'patch' in img_name:
                makedir(pjoin(cfg.data_path, dest_folder, f'class_idx_{class_idx}'))
                shutil.copyfile(src=pjoin(source_folder, img_name),
                                dst=pjoin(cfg.data_path, dest_folder,
                                          f'class_idx_{class_idx}', img_name))
            else:
                prepare_cut_out_image(pjoin(source_folder, img_name), int(class_idx), type)


if __name__ == '__main__':
    seed = 624
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='strategy')
    parser_image = subparser.add_parser('single')
    parser_image.add_argument('path_to_image', type=str)
    parser_image.add_argument('-c', type=int, help='prototype class idx')
    parser_image.add_argument('-dest', type=str, choices=['confound', 'remember'],
                              default='confound')

    parser_int = subparser.add_parser('interactive',
                                      description='extract activated patches from the '
                                                  'most activated images of each prototype '
                                                  'and choose which must be kept or be forgotten')
    parser_int.add_argument('path_to_folder', type=str,
                            help='path to the folder containing the most activated patches'
                                 'for each prototype')
    parser_int.add_argument('-classes', nargs='+', type=int, default=None,
                            help='the (0-based) index of the classes whose prototypes you want to debug')
    parser_int.add_argument('-n-img', type=int, default=10,
                            help='number of nearest patches to show for each prototype')

    parser_move = subparser.add_parser('move',
                                       description='copy patches to remember or '
                                                   'to forget to the data set folder')
    parser_move.add_argument('path_to_model_dir', type=str)

    args = parser.parse_args()

    if args.strategy == 'single':
        prepare_cut_out_image(args.path_to_image, args.c, args.dest)
    elif args.strategy == 'interactive':
        cfg = OmegaConf.create(
            load(os.path.join(args.path_to_folder, '..', 'stat.pickle'))['cfg'])['data']
        patches_extraction(args.path_to_folder, args.classes, args.n_img)
        interactive_debugging(args.path_to_folder)
    elif args.strategy == 'move':
        cfg = OmegaConf.create(
            load(os.path.join(args.path_to_model_dir, 'stat.pickle'))['cfg'])
        cfg = settings.DATASET_CONFIGS[cfg.data.name]
        move_patches_to_forbidden_remember_folder(args.path_to_model_dir)
