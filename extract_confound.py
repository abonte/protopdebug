import argparse
import os
import re
import shutil
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from omegaconf.omegaconf import OmegaConf

import settings
from helpers import makedir, load


def extract_confound(path_to_image, class_idx, dest, random_bg=False):
    size_in_original_img_of_one_pixel_in_latent_space = (32, 32)
    img = Image.open(os.path.join(path_to_image))
    background = _prepare_forbidden_confound(cfg.img_size, img, random_bg)
    _save_image_to_forbidden_folder(cfg, class_idx, background, dest)


def _prepare_forbidden_confound(img_size, img, random_bg):
    # TODO fix to right receptive field
    # img = T.CenterCrop(size=size_in_original_img_of_one_pixel_in_latent_space)(img)
    img_w, img_h = img.size
    if random_bg:
        imarray = np.random.rand(img_size, img_size, 3) * 255
        background = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    else:
        background = Image.new(mode="RGB",
                               size=(img_size, img_size),
                               color='white')
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    return background


def _save_image_to_forbidden_folder(cfg, cls_idx: int, image: Image, dest: str,
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


def interactive_confound_extraction(path: str,
                                    selected_classes_idx: list,
                                    n_images_to_show: int =5,
                                    apr_threshold: float=0.5):
    # selected_classes_idx = [8, 4, 5, 11, 1]
    # selected_classes_idx = [0,8,14,6,15]
    if selected_classes_idx is not None:
        print(f'Warning: show only {selected_classes_idx}')
    img_class_idxs = np.load(os.path.join(path, "full_class_id.npy"))
    print(img_class_idxs.shape)
    proto_class_identity = \
        load(os.path.join(os.path.dirname(path), 'stat.pickle'))['proto_identity']

    n_prototypes, _ = img_class_idxs.shape

    max_k = min(img_class_idxs.shape[1] + 1, n_images_to_show+1)
    yes, no, p, r = 0, 0, 0, 0
    conf_dest_folder = pjoin(path, '..', 'tmp_forbidden_conf')
    remember_dest_folder = pjoin(path, '..', 'tmp_remember_patch')
    makedir(conf_dest_folder)
    makedir(remember_dest_folder)

    for proto_idx, row in zip(range(0, n_prototypes), img_class_idxs):
        proto_class_idx = np.where(proto_class_identity[proto_idx] == 1)[0][0]
        if selected_classes_idx is not None:
            if proto_class_idx not in selected_classes_idx:
                continue
        for k in range(1, max_k):
            act_filename = pjoin(path, str(proto_idx), f"nearest-{k}_act.pickle")
            if not os.path.exists(act_filename):
                continue
            act, act_pr = load(
                pjoin(path, str(proto_idx), f"nearest-{k}_act.pickle"))

            fig, axs = plt.subplots(1, 3)
            axs[0].set_title(
                f'img_cls=({row[k - 1]},{proto_class_idx}) p={proto_idx} i={k}\n'
                f'max_act={np.max(act):.2f}\n'
                f'mean_act={np.mean(act):.2f}\n'
                f'apr={act_pr if act_pr is None else round(act_pr, 2)}')
            img_act_map = plt.imread(pjoin(path, str(proto_idx),
                                           f'nearest-{k}_original_with_heatmap.png'))
            axs[0].imshow(img_act_map)
            axs[0].axis('off')

            patch_on_image = plt.imread(pjoin(path, str(proto_idx),
                                              f'nearest-{k}_high_act_patch_in_original_img.png'))
            axs[1].imshow(patch_on_image)
            axs[1].axis('off')

            patch_path = pjoin(path, str(proto_idx), f'nearest-{k}_high_act_patch.png')
            axs[2].imshow(plt.imread(patch_path))
            axs[2].axis('off')

            plt.show(block=False)

            save_name = f'c={proto_class_idx}_p={proto_idx}_i={k}'

            while True:
                select = input(f"select image c={proto_class_idx} i={k}? [y (conf)|n (next)|p (conf patch) |r (remember)] ")
                if select == 'y':
                    yes += 1
                    shutil.copy(src=pjoin(path, str(proto_idx),
                                          f'nearest-{k}_original.png'),
                                dst=pjoin(conf_dest_folder, save_name+'.png'))
                    shutil.copy(src=pjoin(path, str(proto_idx),
                                          f'nearest-{k}_original_with_heatmap.png'),
                                dst=pjoin(conf_dest_folder, f'{save_name}_act.png'))
                    break
                elif select == 'n':
                    no += 1
                    break
                elif select == 'p':
                    p += 1
                    patch = Image.open(patch_path)
                    forbidden_conf = _prepare_forbidden_confound(cfg.img_size,
                                                                 patch, False)
                    forbidden_conf.save(pjoin(conf_dest_folder, save_name+'patch.png'))
                    shutil.copy(src=pjoin(path, str(proto_idx),
                                          f'nearest-{k}_original_with_heatmap.png'),
                                dst=pjoin(conf_dest_folder, f'{save_name}_act.png'))
                    break
                elif select == 'r':
                    r += 1
                    shutil.copy(src=pjoin(path, str(proto_idx),
                                          f'nearest-{k}_original.png'),
                                dst=pjoin(remember_dest_folder, save_name + '.png'))
                    shutil.copy(src=pjoin(path, str(proto_idx),
                                          f'nearest-{k}_original_with_heatmap.png'),
                                dst=pjoin(remember_dest_folder, f'{save_name}_act.png'))
                    break
                else:
                    print(f'Wrong value {select}')
                    continue
            plt.close(fig)
    print(f'Stats: yes={yes} no={no} p={p} remember={r}')


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
                                dst=pjoin(cfg.data_path, dest_folder, f'class_idx_{class_idx}', img_name))
            else:
                extract_confound(pjoin(source_folder, img_name), int(class_idx), type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='strategy')
    parser_image = subparser.add_parser('single')
    parser_image.add_argument('path_to_image', type=str)
    parser_image.add_argument('-c', type=int, help='prototype class idx')
    parser_image.add_argument('-dest', type=str, choices=['confound', 'remember'],
                              default='confound')

    parser_int = subparser.add_parser('interactive')
    parser_int.add_argument('path_to_folder', type=str,
                            help='path to the folder containing the most activated patches'
                                 'for each prototype')
    parser_int.add_argument('-classes', nargs='+', type=int, default=None,
                            help='the (0-based) index of the classes whose prototypes you want to debug')
    parser_int.add_argument('-n-img', type=int, default=10,
                            help='number of nearest patches to show for each prototype')

    parser_move = subparser.add_parser('move')
    parser_move.add_argument('path_to_model_dir', type=str)

    args = parser.parse_args()

    if args.strategy == 'single':
        extract_confound(args.path_to_image, args.c, args.dest)
    elif args.strategy == 'interactive':
        cfg = OmegaConf.create(load(os.path.join(args.path_to_folder, '..', 'stat.pickle'))['cfg'])['data']
        interactive_confound_extraction(args.path_to_folder, args.classes, args.n_img)
    elif args.strategy == 'move':
        cfg = OmegaConf.create(load(os.path.join(args.path_to_model_dir, 'stat.pickle'))['cfg'])
        cfg = settings.DATASET_CONFIGS[cfg.data.name]
        move_patches_to_forbidden_remember_folder(args.path_to_model_dir)
