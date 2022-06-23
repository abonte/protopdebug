import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from hydra.utils import to_absolute_path
from omegaconf.omegaconf import MISSING

ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
# path to the CUB 200-2011 folder (e.g., /path/to/CUB_200_2011)
PATH_TO_RAW_CUB_200 = 'datasets/CUB_200_2011'


@dataclass
class BaseDataset:
    """
    data_path: str
        path to the root folder of the dataset (e.g., datasets/cub200_cropped)
    train_dir: str
        directory containing the augmented training set (<data_path>/train_cropped_augmented)
    test_dir: str
        the directory containing the test set (<data_path>/test_cropped")
    train_push_dir: str
        the directory containing the original (unaugmented) training set (<data_path>/train_cropped)
    """
    name: str = MISSING
    img_size: int = MISSING
    num_classes: int = MISSING
    data_path: str = MISSING
    relative_data_path: str = MISSING
    train_directory: str = 'train_cropped_augmented'
    train_push_directory: str = 'train_cropped'
    test_directory: str = 'test_cropped'
    remembering_protos_directory: str = 'remembering_prototypes'
    forbidden_protos_directory: str = 'forbidden_prototypes'
    train_batch_size: int = MISSING
    test_batch_size: int = MISSING
    train_push_batch_size: int = MISSING
    test_on_segmented_image: bool = False
    train_on_segmented_image: bool = False


@dataclass
class Cub200DatasetConfig(BaseDataset):
    img_size: int = 224
    train_batch_size: int = 20
    test_batch_size: int = 30
    train_push_batch_size: int = 25


@dataclass
class Cub200ArtificialConfound(Cub200DatasetConfig):
    name: str = 'cub200_artificial'
    num_classes: int = 5
    relative_data_path: str = 'datasets/cub200_cropped/confound_artificial'
    data_path: str = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                  relative_data_path)


@dataclass
class Cub200Clean5(Cub200DatasetConfig):
    name: str = 'cub200_clean_5'
    num_classes: int = 5
    relative_data_path: str = 'datasets/cub200_cropped/clean_5_classes'
    data_path: str = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                  relative_data_path)


@dataclass
class Cub200CleanTop20(Cub200DatasetConfig):
    name: str = 'cub200_clean_top20'
    num_classes: int = 20
    train_batch_size: int = 128
    relative_data_path: str = 'datasets/cub200_cropped/clean_top_20'
    test_directory = 'test_cropped_shuffled'
    data_path: str = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                  relative_data_path)


@dataclass
class Cub200CleanAll(Cub200DatasetConfig):
    name: str = 'cub200_clean_all'
    num_classes: int = 200
    train_batch_size = 80
    test_batch_size = 100
    train_push_batch_size = 75
    relative_data_path: str = 'datasets/cub200_cropped/clean_all'
    data_path: str = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                  relative_data_path)


@dataclass
class Cub200BackgroundConfound(Cub200DatasetConfig):
    name: str = 'cub200_bgconf'
    relative_data_path: str = 'datasets/cub200_cropped/confound_background'
    data_path: str = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                  relative_data_path)


@dataclass
class SyntheticDatasetConfig(BaseDataset):
    name: str = 'synthetic'
    data_path = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                             'datasets/synthetic')
    img_size = 80
    num_classes = 3
    train_batch_size = 10
    test_batch_size = 25
    train_push_batch_size = 30


@dataclass
class CovidDatasetConfig(BaseDataset):
    name: str = 'covid'
    relative_data_path: str = 'datasets/covid'
    data_path = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                             relative_data_path)
    img_size: int = 224
    num_classes = 2
    train_batch_size = 20
    test_batch_size = 30
    train_push_batch_size = 35


DATASET_CONFIGS = {
    Cub200ArtificialConfound.name: Cub200ArtificialConfound,
    Cub200Clean5.name: Cub200Clean5,
    Cub200CleanTop20.name: Cub200CleanTop20,
    Cub200CleanAll.name: Cub200CleanAll,
    Cub200BackgroundConfound.name: Cub200BackgroundConfound,
    SyntheticDatasetConfig.name: SyntheticDatasetConfig,
    CovidDatasetConfig.name: CovidDatasetConfig
}


@dataclass
class ModelConfig:
    base_architecture: str = 'vgg11'
    pretrained_model_path: str = to_absolute_path('pretrained_models')
    num_prototypes_per_class: int = 2
    prototype_shape: tuple = (128, 1, 1)
    prototype_activation_function: str = 'log'
    add_on_layers_type: str = 'regular'

    joint_optimizer_lrs: Dict[str, float] = field(default_factory=
                                                  lambda: {'features': 1e-4,
                                                           'add_on_layers': 3e-3,
                                                           'prototype_vectors': 3e-3})
    joint_lr_step_size: int = 5

    warm_optimizer_lrs: Dict[str, float] = field(default_factory=
                                                 lambda: {'add_on_layers': 3e-3,
                                                          'prototype_vectors': 3e-3})

    last_layer_optimizer_lr: float = 1e-4
    gamma: float = 0.15  # original 0.1 Gamma of learning rate scheduler of the joint optimizer

    @dataclass
    class Lambdas:
        # coefs: weighting of different training losses
        crs_ent: float = 1
        clst: float = 0.5
        sep: float = -0.08
        l1: float = 1e-4
        debug: float = 0
        rem: float = 0
        # clst_step_size: int = 1
        # clst_gamma: float = 1.001

    coefs: Lambdas = Lambdas()

    # 1 as in original PNet paper, 5% in IAIA-BL paper
    # if value in (0,1), relative number of patches
    # If >= 1, absolute number of patches
    topk_k: float = 1


@dataclass
class Debug:
    # type of loss used to fix wrong prototype
    # ['attribute', 'aggregation', 'prototypes', 'kernel', 'iaiabl']
    loss: Optional[str] = None
    fine_annotation: Optional[float] = None  # % of images that are fine annotated
    path_to_model: Optional[str] = None  # load existing model
    load_optimizer: bool = False  # load optimizer and scheduler from path_to_model
    auto_path_to_model: bool = False  # take model from previous iteration
    hard_constraint: bool = False
    # list of prototype number to remove [0, number of prototypes)
    protos: List[int] = field(default_factory=list)
    #  list of classes idx [0,number of classes)
    classes: List[int] = field(default_factory=list)
    # only for aggregation loss
    class_specific_penalization: bool = True
    act_place: str = 'center'  # ['center', 'all']
    epsilon: float = 1e-8


# defaults = [
#    {"data": "cub200_clean_top20"}
# ]


@dataclass
class ExperimentConfig:
    """
        gpuid: list
            GPU device ID(s)
        wandb: bool
        warm_epochs: int
            number of epochs (passes through the dataset)
        last_layer_iterations: int
            number of last layer optimization iterations, before pushing the prototypes
        push_start: int
            push start epoch
    """
    # defaults: List[Any] = field(default_factory=lambda: defaults)

    experiment_name: str = MISSING
    dry_run: bool = False
    seed: int = 0
    cpu: bool = False
    gpuid: List[str] = field(default_factory=lambda: [0, 1])
    wandb: bool = False
    epochs: int = 15
    warm_epochs: int = 5
    last_layer_iterations: int = 0
    push_start: int = 0
    push_epochs: Optional[List[int]] = field(default_factory=list)

    data: BaseDataset = MISSING
    model: ModelConfig = ModelConfig()
    debug: Debug = Debug()
