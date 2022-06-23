#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

import matplotlib.pyplot as plt
import glob
import numpy as np
from omegaconf.omegaconf import OmegaConf
import matplotlib as mpl


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def plot(filename, plotname, n_classes, n_prototypes_per_class, colorbar=False):
    print(filename)
    figsize = (n_prototypes_per_class * 3.4, n_classes * 3.4)
    fig, axs = plt.subplots(n_classes, n_prototypes_per_class, figsize=figsize)

    for i, ax in enumerate(axs.flatten()):
        act = np.load(os.path.join(path, f"prototype-self-act{i}.npy"))
        ax.set_title(f'class={int(i / n_prototypes_per_class)} proto_idx={i}\n'
                     f'max_act={np.max(act):.2f}\n'
                     f'mean_act={np.mean(act):.2f}'
                     # f'\nmedian_act={np.median(act):.2f}'
                     )

        if colorbar:
            cmap = plt.cm.get_cmap("jet")
            norm = mpl.colors.Normalize(vmin=np.min(act), vmax=np.max(act))
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8,
                         format='%.2f')

        image = plt.imread(os.path.join(path, filename.format(i)))
        ax.imshow(image)
        ax.axis('off')

    plt.savefig(plotname, bbox_inches='tight')


def plot_subset_of_classes(filename, plotname, n_prototypes_per_class, cls,
                           colorbar=False):
    print(filename)
    # cls_idx = [200-1, 46-1, 183-1, 26-1, 86-1, 4-1, 175-1, 1-1, 188-1, 40-1]
    # cls = [8, 0, 1, 4, 7]
    # cls = [8,4,5,11,1]
    # cls = [0, 8, 14, 6, 15]
    cls.sort()
    n_classes = len(cls)
    figsize = (n_prototypes_per_class * 3.4, n_classes * 3.4)
    fig, axs = plt.subplots(n_classes, n_prototypes_per_class, figsize=figsize)
    for e, cl in enumerate(cls):
        for i_tmp in range(2):
            ax = axs[e, i_tmp]
            i = i_tmp + (cl * n_prototypes_per_class)

            act = np.load(os.path.join(path, f"prototype-self-act{i}.npy"))
            ax.set_title(f'class={cl} proto_idx={i}\n'
                         f'max_act={np.max(act):.2f}\n'
                         f'mean_act={np.mean(act):.2f}')

            if colorbar:
                cmap = plt.cm.get_cmap("jet")
                norm = mpl.colors.Normalize(vmin=np.min(act), vmax=np.max(act))
                fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                             shrink=0.8,
                             format='%.2f')

            image = plt.imread(os.path.join(path, filename.format(i)))
            ax.imshow(image)
            ax.axis('off')

    plt.savefig(plotname, bbox_inches='tight')


def main(path, n_classes, n_prototypes, epoch, classes):
    if classes is not None:
        plot_subset_of_classes('prototype-img-original_with_self_act{0}.png',
                               os.path.join(path, '..',
                                            f'top_5_epoch_{epoch}_all_prototypes_act.png'),
                               n_prototypes, classes, colorbar=True)

        plot_subset_of_classes('prototype-img-original{0}.png',
                               os.path.join(path, '..',
                                            f'top_5_epoch_{epoch}_all_prototypes_original.png'),
                               n_prototypes, classes)
    else:
        plot('prototype-img-original_with_self_act{0}.png',
             os.path.join(path, '..', f'epoch_{epoch}_all_prototypes_act.png'),
             n_classes,
             n_prototypes, colorbar=True)

        plot('prototype-img-original{0}.png',
             os.path.join(path, '..', f'epoch_{epoch}_all_prototypes_original.png'),
             n_classes, n_prototypes)

        # plot('prototype-img-receptive_field_with_self_act{0}.png',
        #      os.path.join(path, '..',
        #                   f'epoch_{epoch}_all_prototypes_receptive_field_act.png'),
        #      n_classes,
        #      n_prototypes)

        # plot('prototype-img-receptive_field{0}.png',
        #      os.path.join(path, '..',
        #                   f'epoch_{epoch}_all_prototypes_receptive_field_original.png'),
        #      n_classes, n_prototypes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-modeldir', help='Example: ../saved_models/vgg11/007',
                        type=str)
    parser.add_argument('-epoch', type=int, default=None)
    parser.add_argument('-classes', nargs='+', type=int, help='classes index',
                        default=None)
    args = parser.parse_args()

    stat = OmegaConf.create(load(os.path.join(args.modeldir, 'stat.pickle'))['cfg'])

    if args.epoch is None:
        path = sorted(glob.glob(os.path.join(args.modeldir, 'img/epoch-*')))[-1]
        epoch = int(os.path.basename(path).split('-')[-1])
    else:
        path = os.path.join(args.modeldir, f'img/epoch-{args.epoch}')
        epoch = args.epoch

    main(path, stat.data.num_classes, stat.model.num_prototypes_per_class, epoch,
         args.classes)
