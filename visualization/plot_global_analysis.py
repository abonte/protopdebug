#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import argparse


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def plot_grid(path, image_name_pattern, output_filename, proto_class_identity, selected_classes_idx):
    # selected_classes_idx = [0, 1, 4, 7, 8]
    # selected_classes_idx = [8,4,5,11,1]
    # selected_classes_idx = [0,8,14,6,15]
    if selected_classes_idx is not None:
        print(f'Warning: show only {selected_classes_idx}')
    img_class_idxs = np.load(os.path.join(path, "full_class_id.npy"))
    print(img_class_idxs.shape)

    n_prototypes, n_img = img_class_idxs.shape
    n_rows = len(selected_classes_idx) * 2 if selected_classes_idx is not None else n_prototypes

    # n_img = 100
    #n_rows = 12
    figsize = (n_img * 3.0, n_rows * 2.5)

    fig, axs = plt.subplots(n_rows, n_img, figsize=figsize)

    i = -1
    max_k = img_class_idxs.shape[1] + 1
    for proto_idx, row in zip(range(0, n_prototypes), img_class_idxs):
        proto_class_idx = np.where(proto_class_identity[proto_idx] == 1)[0][0]
        if selected_classes_idx is not None:
            if proto_class_idx not in selected_classes_idx:
                continue
        i += 1
        e = -1
        for k in range(1, max_k):
            # e += 1
            # if k % 100 == 0 and k !=0 :
            #     i += 1
            #     e = 0
            e = k -1
            act_filename = os.path.join(path, str(proto_idx),
                                        f"nearest-{k}_act.pickle")
            if not os.path.exists(act_filename):
                continue
            ax = axs[i, e]
            act, act_pr = load(
                os.path.join(path, str(proto_idx), f"nearest-{k}_act.pickle"))
            image = plt.imread(os.path.join(path, str(proto_idx),
                                            image_name_pattern.format(k)))

            ax.set_title(f'img_cls=({row[k - 1]},{proto_class_idx}) p={proto_idx} i={k}\n'
                         f'max_act={np.max(act):.4f}\n'
                         f'mean_act={np.mean(act):.4f}\n'
                         f'apr={act_pr if act_pr is None else round(act_pr, 2)}')
            ax.imshow(image)
            ax.axis('off')

    fig.tight_layout()

    plt.savefig(os.path.join(path, output_filename + '.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to model')
    parser.add_argument('-classes', nargs='+', type=int, help='classes index', default=None)
    args = parser.parse_args()

    exp = args.model.split('.pth')[0]

    for path in [exp + "_nearest_train",
                 exp + "_nearest_test",
                 exp + '_nearest_kernel_set'
                 ]:
        print(path)
        if not os.path.exists(path):
            continue

        # (n_prototypes, n_classes)
        # try:
        proto_class_identity = \
            load(os.path.join(os.path.dirname(args.model), 'stat.pickle'))['proto_identity']
        # except KeyError:
        # proto_class_identity = np.load(os.path.join(os.path.dirname(args.model), 'proto_class_identity.npy'))

        plot_grid(path, 'nearest-{}_original_with_heatmap.png', 'heatmap_'+os.path.basename(path),
                  proto_class_identity, args.classes)
        plot_grid(path, 'nearest-{}_high_act_patch_in_original_img.png', 'original_'+os.path.basename(path),
                  proto_class_identity, args.classes)
