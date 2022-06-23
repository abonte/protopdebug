#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw
from PIL.ImageColor import getrgb
from omegaconf.omegaconf import OmegaConf

import settings
from settings import SyntheticDatasetConfig

transpose = [
    Image.FLIP_LEFT_RIGHT,
    Image.FLIP_TOP_BOTTOM,
    # Image.ROTATE_90,
    Image.ROTATE_180,
    # Image.ROTATE_270,
    # Image.TRANSPOSE
]

shapes = ['ellipse',
          'rectangle',
          'triangle'
          # 'line'
          ]

shape_colors = ['black', 'red', 'green', 'blue', 'yellow', 'orange', 'purple']
# background_colors = shape_colors + ['white']
background_colors = ['white']

n_max_shape_in_img = 1
n_min_shape_in_img = 1
n_times_image_transposed = 0
extension = 'png'


#    (((Upper left x coordinate, upper left y coordinate), (lower right x coordinate, lower right y coordinate))
#    (Upper left x coordinate, upper left y coordinate, lower right x coordinate, lower right y coordinate)

def _plot_shape(draw, img_size, exclude_color=None, shape=None, color=None,
                upper_left_xy=None,
                return_coordinates=False, loc: str = 'random'):
    if shape is None:
        shape = random.choice(shapes)
    if color is None:
        while True:
            color = random.choice(shape_colors)
            if color != exclude_color:
                break

    fill = getrgb(color)
    if loc == 'random':
        if upper_left_xy is None:
            upper_left_x = random.randint(1, int(img_size * 0.85))  # int(width/2))
            upper_left_y = random.randint(1, int(img_size * 0.85))  # int(height/2))
        else:
            upper_left_x, upper_left_y = upper_left_xy

        # lower_right_x = min(upper_left_x + random.randint(10, int(width/3)), width-1)
        # lower_right_y = min(upper_left_y + random.randint(10, int(height/3)), height-1)

        # points = [(upper_left_x, upper_left_y), (lower_right_x, lower_right_y)]

        if shape == 'ellipse':
            lower_right_x = min(upper_left_x + 5, img_size - 1)
            lower_right_y = min(upper_left_y + 5, img_size - 1)
            points = [(upper_left_x, upper_left_y), (lower_right_x, lower_right_y)]
            draw.ellipse(points, fill=fill)
        elif shape == 'rectangle':
            lower_right_x = min(upper_left_x + 5, img_size - 1)
            lower_right_y = min(upper_left_y + 5, img_size - 1)

            points = [(upper_left_x, upper_left_y), (lower_right_x, lower_right_y)]
            draw.rectangle(points, fill=fill)
        elif shape == 'triangle':
            points = [(upper_left_x, upper_left_y),
                      (upper_left_x + 5, upper_left_y + 5),
                      (upper_left_x + 10, upper_left_y)]

            draw.polygon(points, fill=fill)
        else:
            raise ValueError()
    elif loc == 'center':
        # central pixel coordinates
        pixel_side_length = 5  # img_size / 7
        upper_left_x = (img_size / 7) * 3 - 1
        upper_left_y = (img_size / 7) * 3 - 1
        lower_right_x = upper_left_x + pixel_side_length
        lower_right_y = upper_left_y + pixel_side_length

        if shape == 'ellipse':
            points = [(upper_left_x, upper_left_y), (lower_right_x, lower_right_y)]
            draw.ellipse(points, fill=fill)
        elif shape == 'rectangle':
            points = [(upper_left_x, upper_left_y),
                      (lower_right_x, lower_right_y)]
            draw.rectangle(points, fill=fill)
        elif shape == 'triangle':

            points = [(upper_left_x, upper_left_y),
                      (upper_left_x + pixel_side_length, upper_left_y),
                      (upper_left_x + pixel_side_length / 2,
                       upper_left_y + pixel_side_length)]

            draw.polygon(points, fill=fill)
        else:
            raise ValueError()

    if return_coordinates:
        return points


def draw_image(img_size):
    background_color = random.choice(background_colors)

    img = Image.new(mode="RGB", size=(img_size, img_size),
                    color=getrgb(background_color))
    draw = ImageDraw.Draw(img)

    list_of_figures = []
    for k in range(random.randint(n_min_shape_in_img, n_max_shape_in_img)):
        s = random.choice(shapes)
        while True:
            c = random.choice(shape_colors)
            if c != background_color:
                break

        list_of_figures.append((s, c))
        _plot_shape(draw, img_size, shape=s, color=c)

    return img, list_of_figures


def return_class(classes, formula, list_of_figures):
    for c, f in zip(classes, formula):
        for figure in list_of_figures:
            if figure in f:
                return c


def all_possible_shapes(path_to_dataset, img_size):
    p = os.path.join(path_to_dataset, 'confounder', 'kernel_test', '000')
    if not os.path.exists(p):
        os.makedirs(p)
    i = 0
    for s in shapes:
        for c in shape_colors:
            img = Image.new(mode="RGB", size=(img_size, img_size), color='white')
            draw = ImageDraw.Draw(img)
            _plot_shape(draw, img_size, shape=s, color=c,
                        upper_left_xy=(img_size / 2, img_size / 2))
            img.save(os.path.join(p, f'{i}_kernel_test.{extension}'), quality=95)
            i += 1


def images_to_compute_loss(path_to_dataset, img_size):
    p = os.path.join(path_to_dataset, 'confounder', 'loss_image', 'class_idx_0')
    os.makedirs(p) if not os.path.exists(p) else ''
    i = 0
    s = 'triangle'
    c = 'red'
    for _ in range(1):
        # for bg in background_colors:
        # imarray = np.random.rand(img_size, img_size, 3) * 255
        # img = Image.fromarray(imarray.astype('uint8')).convert('RGB')

        img = Image.new(mode="RGB", size=(img_size, img_size), color=getrgb('white'))
        draw = ImageDraw.Draw(img)

        # tt = 0
        # print(img_size)
        # for ll in range(3, img_size+1, 4):
        #     draw.line([(ll,0), (ll, img_size)], fill=getrgb('black'))
        #     draw.line([(0, ll), (img_size, ll)], fill=getrgb('black'))

        _plot_shape(draw, img_size, shape=s, color=c, loc='center')
        img.save(os.path.join(p, f'{i}_kernel_test.{extension}'), quality=95)
        i += 1


def generate_train_test_synthetic_dataset(path_to_dataset, img_size):
    random.seed(2)
    # or condition
    formulas = [
        [('rectangle', 'blue'), ('ellipse', 'green')],
        [('ellipse', 'yellow')],
        [('triangle', 'black')]
    ]

    cofounder = {'000': ('triangle', 'red')}

    assert len(formulas) <= len(shape_colors)
    n_classes = len(formulas)
    n_train_images = 80 * n_classes
    n_test_images = 80 * n_classes

    # create folders
    # push directory and augmented directory are the same
    test_path_clean = os.path.join(path_to_dataset,
                                   'clean',
                                   settings.url['synthetic']['test_directory'])
    test_path_confound = os.path.join(path_to_dataset,
                                      'confound',
                                      settings.url['synthetic']['test_directory'])
    train_aug_path_clean = os.path.join(path_to_dataset,
                                        'clean',
                                        settings.url['synthetic'][
                                            'train_push_directory'])
    train_aug_path_confound = os.path.join(path_to_dataset,
                                           'confound',
                                           settings.url['synthetic'][
                                               'train_push_directory'])
    train_aug_mask_path = os.path.join(path_to_dataset,
                                       'confound',
                                       settings.url['synthetic'][
                                           'train_push_directory'] + '_mask')

    classes = []
    for w in range(len(formulas)):
        cl_name = str(w).zfill(3)
        classes.append(cl_name)
        for p in [test_path_clean, test_path_confound, train_aug_path_clean,
                  train_aug_path_confound, train_aug_mask_path]:
            p = os.path.join(p, cl_name)
            if not os.path.exists(p):
                os.makedirs(p)

    print('generate training test')
    i = 0
    while i <= n_train_images:
        img, list_of_figures = draw_image(img_size)
        cl = return_class(classes, formulas, list_of_figures)
        if cl is None or len(
                glob.glob(os.path.join(train_aug_path_confound, cl, '*'))) > (
                n_train_images / len(formulas)):
            completed = all(
                [len(glob.glob(os.path.join(train_aug_path_confound, cc, '*'))) > (
                        n_train_images / len(formulas)) for cc in classes])
            if completed:
                break
            continue

        image_name = f'{i}_class_{cl}.{extension}'
        img.save(os.path.join(train_aug_path_clean, cl, image_name),
                 quality=95)

        # add confounder
        if cl in cofounder.keys():
            if random.random() <= 1:
                draw = ImageDraw.Draw(img)
                coordinates = _plot_shape(draw, img_size,
                                          shape=cofounder[cl][0],
                                          color=cofounder[cl][1],
                                          return_coordinates=True)

                # save mask
                mask = Image.new(mode="1",
                                 size=(img_size, img_size),
                                 color=1)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.polygon(coordinates, fill=0)
                mask.save(os.path.join(train_aug_mask_path, cl, image_name))

        img.save(os.path.join(train_aug_path_confound, cl, image_name), quality=95)
        i += 1

    print('generate test')
    i = 0
    while i <= n_test_images:
        img, list_of_figures = draw_image(img_size)
        cl = return_class(classes, formulas, list_of_figures)
        if cl is None or len(glob.glob(os.path.join(test_path_confound, cl, '*'))) > (
                n_test_images / len(formulas)):
            completed = all([len(glob.glob(os.path.join(test_path_clean, cc, '*'))) > (
                    n_test_images / len(formulas)) for cc in classes])
            if completed:
                break
            continue

        img.save(os.path.join(test_path_clean, cl, f'{i}_class_{cl}.{extension}'),
                 quality=95)
        img.save(os.path.join(test_path_confound, cl, f'{i}_class_{cl}.{extension}'),
                 quality=95)
        i += 1

    # count example in each class
    for phase, phase_path in [('Train', train_aug_path_confound),
                              ('Test', test_path_confound)]:
        print(phase)
        total = 0
        for cl in classes:
            n = len(glob.glob(os.path.join(phase_path, cl, '*')))
            total += n
            print('\t', 'class=', cl, 'n=', n)
        print('total', total)


def generate_synthetic_dataset(data_conf: SyntheticDatasetConfig, seed: int):
    random.seed(seed)
    train_dir = os.path.join(data_conf.data_path, data_conf.train_directory)
    path_to_dataset = os.path.join(train_dir, '..')

    all_possible_shapes(path_to_dataset, data_conf.img_size)
    images_to_compute_loss(path_to_dataset, data_conf.img_size)
    generate_train_test_synthetic_dataset(path_to_dataset, data_conf.img_size)


if __name__ == "__main__":
    conf: SyntheticDatasetConfig = OmegaConf.structured(SyntheticDatasetConfig)
    generate_synthetic_dataset(data_conf=conf, seed=0)
    print('Done!')
