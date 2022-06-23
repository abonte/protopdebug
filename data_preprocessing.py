import argparse
import os.path

from omegaconf.omegaconf import OmegaConf

from data.bird_add_confound import add_confounds
from data.bird_initial_images_processing import crop_and_train_test_split, augmentation
from data.generate_synthetic_images import generate_synthetic_dataset
from data.shuffle_test_background import shuffle_bg
from settings import SyntheticDatasetConfig, \
    Cub200Clean5, Cub200ArtificialConfound, Cub200CleanAll, PATH_TO_RAW_CUB_200

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0)
    subparser = parser.add_subparsers(title='Dataset', dest='dataset_name')

    parser_bird = subparser.add_parser('cub200', help='bird CUB_200_2011 dataset')
    parser_bird.add_argument('--classes', nargs='+', type=str, default=None,
                             help='space separated list of classes to process (e.g., 001 002 003), default '
                                  'all classes')

    parser_bird = subparser.add_parser('cub200_shuffled_bg',
                                       help='preprocess all classes of bird CUB_200_2011 dataset and shuffle backgrounds')

    parser_synthetic = subparser.add_parser('synthetic',
                                            help='synthetic dataset with various shapes (3 classes)')

    parser_covid = subparser.add_parser('covid')

    args = parser.parse_args()

    if args.dataset_name == 'cub200':
        images_dir = os.path.abspath(PATH_TO_RAW_CUB_200)
        clean_conf: Cub200Clean5 = OmegaConf.structured(Cub200Clean5)
        train_push_dir = os.path.join(clean_conf.data_path,
                                      clean_conf.train_push_directory)
        train_dir = os.path.join(clean_conf.data_path, clean_conf.train_directory)
        test_dir = os.path.join(clean_conf.data_path, clean_conf.test_directory)

        crop_and_train_test_split(images_dir, train_push_dir, test_dir, args.classes)
        augmentation(source_dir=train_push_dir,
                     dest_dir=train_dir,
                     mask_dir=train_push_dir + '_segmentation',
                     seed=args.seed)

        confound_conf: Cub200ArtificialConfound = OmegaConf.structured(
            Cub200ArtificialConfound)
        add_confounds(clean_conf, confound_conf, args.seed)

    elif args.dataset_name == 'cub200_shuffled_bg':
        images_dir = os.path.abspath(PATH_TO_RAW_CUB_200)
        cfg_all: Cub200CleanAll = OmegaConf.structured(Cub200CleanAll)
        train_push_dir = os.path.join(cfg_all.data_path,
                                      cfg_all.train_push_directory)
        train_dir = os.path.join(cfg_all.data_path, cfg_all.train_directory)
        test_dir = os.path.join(cfg_all.data_path, cfg_all.test_directory)

        crop_and_train_test_split(images_dir, train_push_dir, test_dir, classes=None)
        augmentation(source_dir=train_push_dir,
                     dest_dir=train_dir,
                     mask_dir=train_push_dir + '_segmentation',
                     seed=args.seed)

        shuffle_bg(cfg_all)

    elif args.dataset_name == 'synthetic':
        clean_conf: SyntheticDatasetConfig = OmegaConf.structured(
            SyntheticDatasetConfig)
        generate_synthetic_dataset(clean_conf, args.seed)

    else:
        raise ValueError(args.dataset_name)
