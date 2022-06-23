import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import Augmentor
import cv2

from helpers import makedir


def crop_images_with_confound_background(metadata_dir: str, images_dir: str,
                                         path_training: str, path_test: str,
                                         classes: list):
    """dataset is already split, crop images"""

    images_id = open(Path(metadata_dir).joinpath('images.txt'))
    boxes = open(Path(metadata_dir).joinpath('bounding_boxes.txt'))
    train_test = open(Path(metadata_dir).joinpath('train_test_split.txt'))
    source_test_image = os.path.join(images_dir, 'test')
    source_train_image = os.path.join(images_dir, 'train')

    for (line, box, train) in zip(images_id, boxes, train_test):
        # Parse lines from given files e.g. get coordinates box, image path and name and train or test
        coords = list(map(int, map(float, box.split()[1:])))
        image_path = line.split()[1]
        image_folder = image_path.split("/")[0]
        image_name = image_path.split("/")[1]
        is_train = int(train.split()[1])

        if not any([c in image_folder for c in classes]):
            # skip image if not in one of selected classes
            continue

        if is_train:
            dest_path_folder = os.path.join(path_training, image_folder)
            source_path_folder = os.path.join(source_train_image, image_folder)
        else:
            dest_path_folder = os.path.join(path_test, image_folder)
            source_path_folder = os.path.join(source_test_image, image_folder)

        makedir(dest_path_folder)
        image = _load_and_crop_image(os.path.join(source_path_folder, image_name),
                                     coords)
        cv2.imwrite(os.path.join(dest_path_folder, image_name), image)


def crop_and_train_test_split(images_dir: str, path_training: str,
                              path_test: str,
                              classes: Optional[list]):

    if os.path.exists(path_training) and os.path.exists(path_test):
        print('Dataset already cropped')
        return
    print('Crop image and split train and test set')

    path_test_segmentation = path_test + '_segmentation'
    path_train_segmentation = path_training + '_segmentation'

    images_id = open(Path(images_dir).joinpath('images.txt'))
    boxes = open(Path(images_dir).joinpath('bounding_boxes.txt'))
    train_test = open(Path(images_dir).joinpath('train_test_split.txt'))
    folder = Path(images_dir).joinpath("images/")
    segmentation_folder = Path(images_dir).joinpath("segmentations")

    for (line, box, train) in zip(images_id, boxes, train_test):
        # Parse lines from given files e.g. get coordinates box, image path and name and train or test
        coords = list(map(int, map(float, box.split()[1:])))
        image_path = line.split()[1]
        image_folder = image_path.split("/")[0]
        image_name = image_path.split("/")[1]
        is_train = int(train.split()[1])

        if classes is not None:
            if not any([c in image_folder for c in classes]):
                # skip image if not in one of the selected classes
                continue

        image = _load_and_crop_image(os.path.join(folder, image_path), coords)

        # Store image either as train or test image
        if is_train:
            path_folder = os.path.join(path_training, image_folder)
            path_segmentation = os.path.join(path_train_segmentation, image_folder)
        else:
            path_folder = os.path.join(path_test, image_folder)
            path_segmentation = os.path.join(path_test_segmentation, image_folder)

        segmentation = _load_and_crop_image(os.path.join(segmentation_folder, image_path.replace('.jpg', '.png')), coords)

        makedir(path_segmentation)
        cv2.imwrite(os.path.join(path_segmentation, image_name), segmentation)

        makedir(path_folder)
        cv2.imwrite(os.path.join(path_folder, image_name), image)

    # same classes
    assert os.listdir(path_training) == os.listdir(path_test) == os.listdir(
        path_test_segmentation) == os.listdir(path_train_segmentation)
    # same images
    test_seg_imgs_names = [os.path.basename(p) for p in
                           glob.glob(path_test_segmentation + '/*/*')]
    test_imgs_names = [os.path.basename(p) for p in glob.glob(path_test + '/*/*')]
    assert len(test_imgs_names) == len(test_seg_imgs_names)


def _load_and_crop_image(path_to_img: str, coords):
    image = cv2.imread(path_to_img)[
            coords[1]:coords[1] + coords[3],
            coords[0]:coords[0] + coords[2],
            :]
    return image


def augmentation(source_dir: str, dest_dir: str, mask_dir: str, seed: int = 0):
    if os.path.exists(dest_dir):
        print('Dataset already augmented!')
        return

    print('Dataset augmentation')
    makedir(dest_dir)

    for i, folder in enumerate(next(os.walk(source_dir))[1]):
        fd = os.path.join(source_dir, folder)
        tfd = os.path.abspath(os.path.join(dest_dir, folder))
        md = os.path.abspath(os.path.join(mask_dir, folder))
        tmd = os.path.abspath(os.path.join(dest_dir+'_segmentation', folder))

        # rotation
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.ground_truth(md)
        p.set_seed(seed)
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.process()
        del p
        # skew
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.ground_truth(md)
        p.set_seed(seed)
        p.skew(probability=1, magnitude=0.2)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.process()
        del p
        # shear
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.ground_truth(md)
        p.set_seed(seed)
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for _ in range(10):
            p.process()
        del p
        # random_distortion
        # p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        # p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
        # p.flip_left_right(probability=0.5)
        # for i in range(10):
        #    p.process()
        # del p

        # create a separate folder for the segmentations
        for filename in os.listdir(tfd):
            filename = filename.replace('_original', '')
            if filename.startswith('_groundtruth_(1)'):
                makedir(tmd)
                new_filename = filename.replace('_groundtruth_(1)_','')
                os.rename(os.path.join(tfd, filename), os.path.join(tmd, new_filename))

