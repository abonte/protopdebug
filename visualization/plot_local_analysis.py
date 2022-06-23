#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import os
import numpy as np

from helpers import load

def main():
    for cls in os.listdir(BASE_PATH):
        if ".DS_Store" in cls:
            continue
        print(f'Class: {cls}')
        cls_path = os.path.join(BASE_PATH, cls)

        for img in os.listdir(cls_path):
            if ".DS_Store" in img:
                continue
            print(f'    img: {img}')
            fig, axs = plt.subplots(2, n_higly_activated_patches, figsize=(15, 7))

            path = os.path.join(cls_path, img)
            obj = load(os.path.join(path, 'most_activated_prototypes', 'stats.pickle'))
            activation_value = obj['activation_value']
            prototype_class_identity = obj['class_identity']

            prototype_class_identity = np.load(
                os.path.join(path, 'most_activated_prototypes', 'class_identity.npy'),
                allow_pickle=True)
            activation_value = np.load(
                os.path.join(path, 'most_activated_prototypes', 'activation_value.npy'),
                allow_pickle=True)

            axs = axs.flatten()
            counter = 0

            for i in range(1, n_higly_activated_patches + 1):
                filename = f'prototype_activation_map_by_top-{i}_prototype.png'
                image = plt.imread(
                    os.path.join(path, 'most_activated_prototypes', filename))
                ax = axs[counter]
                ax.imshow(image)
                ax.axis('off')
                counter += 1

            last_layer_weight = np.load(os.path.join(path, 'most_activated_prototypes',
                                                     'last_layer_weight.npy'))
            for i, cl, act in zip(range(1, n_higly_activated_patches + 1),
                                  prototype_class_identity, activation_value):
                filename = f'top-{i}_activated_prototype_in_original_pimg.png'
                image = plt.imread(
                    os.path.join(path, 'most_activated_prototypes', filename))
                ax = axs[counter]
                ax.set_title(f'cl={cl}, act_value={act:.2f} w={last_layer_weight[i-1]:.2f}')
                ax.imshow(image)
                ax.axis('off')
                counter += 1

            output_path = os.path.join(PLOT_PATH, cls)
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, img))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='path to model', type=str)
    parser.add_argument('-o', dest='output', type=str, help="Output path")
    args = parser.parse_args()

    BASE_PATH = args.model.split('.pth')[0]
    PLOT_PATH = args.output if args.output is not None else os.path.join(os.path.dirname(args.model), 'local_analysis')
    model =os.path.basename(BASE_PATH)
    n_higly_activated_patches = 4

    main()
