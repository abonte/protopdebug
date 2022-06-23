import glob
import os
import random
import shutil

import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from PIL.ImageColor import getrgb
from omegaconf.omegaconf import OmegaConf
from scipy import signal

import settings
from helpers import makedir

confounds = [(0, '001', 'green'), (1, '002', 'black'), (4, '005', 'purple')]


# (index of the class in the dataset, class, color rectangular confound)


def _add_confounds(path_to_dataset, data_conf: settings.BaseDataset):
    print('Add confounds')
    target_image_size = data_conf.img_size
    resize = transforms.Compose(
        [transforms.Resize(size=(target_image_size, target_image_size))])

    path_training = os.path.join(path_to_dataset, data_conf.train_push_directory)
    path_augmented = os.path.join(path_to_dataset, data_conf.train_directory)

    for fd in [path_augmented, path_training]:
        print(fd)
        for _, cl, confound_color in confounds:
            # find complete name of the class
            class_with_confound = \
                glob.glob(os.path.join(path_to_dataset, fd, cl) + '*')[0].split('/')[-1]
            print('\t' + class_with_confound)
            assert len(glob.glob(os.path.join(path_to_dataset, fd, cl) + '*')) == 1

            path_to_mask = os.path.join(path_to_dataset, fd + '_segmentation',
                                        class_with_confound)
            if not os.path.exists(path_to_mask):
                os.makedirs(path_to_mask)
            path_to_class = os.path.join(path_to_dataset, fd, class_with_confound)

            for image_name in os.listdir(path_to_class):
                if image_name == '.DS_Store':
                    continue
                path_to_image = os.path.join(path_to_class, image_name)
                img = Image.open(path_to_image)
                img = resize(img)

                draw = ImageDraw.Draw(img)
                coordinates = _plot_confound(draw, img.size,
                                             confound_color,
                                             loc='random')
                img.save(path_to_image)

                mask = Image.new(mode="1",
                                 size=(target_image_size, target_image_size),
                                 color=1)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle(coordinates, fill=0)
                mask.save(os.path.join(path_to_mask, image_name))


def _plot_confound(draw, img_size, color, loc='random', segmentation=None):
    # (0, 0) in the upper left corner
    # [(upper_left_x, upper_left_y), (lower_right_x, lower_right_y)]
    confound_side = 25
    if loc == 'random':
        # random
        upper_left_x = random.randint(0, img_size[0])  # int(width/2))
        upper_left_y = random.randint(0, img_size[1])

        if upper_left_x + confound_side < img_size[0]:
            lower_right_x = upper_left_x + confound_side
        else:
            lower_right_x = upper_left_x - confound_side

        if upper_left_y + confound_side < img_size[1]:
            lower_right_y = upper_left_y + confound_side
        else:
            lower_right_y = upper_left_y - confound_side

        points = [(upper_left_x, upper_left_y), (lower_right_x, lower_right_y)]
        draw.rectangle(points, fill=getrgb(color))
    elif loc == 'center':
        # central pixel of the image in latent space
        # coordinates start from zero
        pixel_side_length = img_size[0] / 7
        upper_left_x = (img_size[0] / 7) * 3
        upper_left_y = (img_size[1] / 7) * 3

        points = [(upper_left_x, upper_left_y),
                  (upper_left_x + pixel_side_length, upper_left_y + pixel_side_length)]
        draw.rectangle(points, fill=getrgb(color))

    elif loc == 'background':
        segmentation = np.array(segmentation)
        segmentation = segmentation / 255
        segmentation[segmentation > 0] = 1
        segmentation[segmentation == 0] = -1
        conf = np.ones((confound_side, confound_side))
        max_peak = np.prod(conf.shape)
        c = signal.correlate(segmentation, conf, 'valid')
        candidate_coordinates_x, candidate_coordinates_y = np.where(c == max_peak)

        if len(candidate_coordinates_x) == 0:
            # random placement
            print('random')
            points = _plot_confound(draw, img_size, color,
                                    loc='random',
                                    segmentation=segmentation)
        else:
            i = np.random.choice(range(len(candidate_coordinates_x)))
            points = [(candidate_coordinates_x[i],
                       candidate_coordinates_y[i]),
                      (candidate_coordinates_x[i] + confound_side,
                       candidate_coordinates_y[i] + confound_side)]
            draw.rectangle(points, fill=getrgb(color))

    else:
        raise ValueError(loc)
    return points


def images_to_compute_loss(path_to_dataset, data_conf: settings.BaseDataset, random_bg=False):
    i = 0
    for idx, _, confound_color in confounds:
        p = os.path.join(path_to_dataset, data_conf.forbidden_protos_directory,
                         f'class_idx_{idx}')
        makedir(p)

        if random_bg:
            imarray = np.random.rand(data_conf.img_size, data_conf.img_size, 3) * 255
            img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        else:
            img = Image.new(mode="RGB",
                            size=(data_conf.img_size, data_conf.img_size),
                            color='white')

        draw = ImageDraw.Draw(img)
        _plot_confound(draw, (data_conf.img_size, data_conf.img_size),
                       color=confound_color, loc='center')
        img.save(os.path.join(p, f'{i}_confound.jpg'), quality=95)
        i += 1


def add_confounds(clean_conf: settings.Cub200Clean5,
                  confound_conf: settings.Cub200ArtificialConfound,
                  seed: int = 1):
    makedir(confound_conf.data_path)

    for src_folder in [clean_conf.train_directory,
                       clean_conf.train_push_directory,
                       clean_conf.test_directory,
                       clean_conf.test_directory + '_segmentation'
                       ]:
        print(f'Copy {src_folder}')
        shutil.copytree(src=os.path.join(clean_conf.data_path, src_folder),
                        dst=os.path.join(confound_conf.data_path, src_folder))

    random.seed(seed)
    np.random.seed(seed)

    _add_confounds(confound_conf.data_path, confound_conf)
    images_to_compute_loss(confound_conf.data_path, confound_conf)


if __name__ == '__main__':
    images_to_compute_loss('prova', settings.Cub200ArtificialConfound())
