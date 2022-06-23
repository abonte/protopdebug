import sys
from datetime import datetime
import glob
import json
import logging
import os
import random
import re
import shutil
import time
from typing import Optional

import hydra
import numpy as np
import torch.utils.data
import wandb
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from omegaconf import omegaconf
from omegaconf.omegaconf import OmegaConf
from prettytable.prettytable import PrettyTable
from tqdm.contrib.telegram import tqdm

import model
import push
import settings
import train_and_test as tnt
from data.data_loader import DATASETS
from helpers import makedir, save_model_w_condition, dump
from preprocess import preprocess_input_function
from settings import ExperimentConfig, Debug, ModelConfig, DATASET_CONFIGS
from telegram_notification import send_telegram_notification
from termination_error import TerminationError

log = logging.getLogger(__name__)


def _get_basename(cfg):
    fields = [
        (None, cfg.experiment_name),
        (None, cfg.data.name),
        (None, cfg.model.base_architecture),
        (None, cfg.debug.loss),
        ('fann', cfg.debug.fine_annotation),
        ('crsent', cfg.model.coefs.crs_ent),
        ('λclst', cfg.model.coefs.clst),
        ('λsep', cfg.model.coefs.sep),
        ('λl1', cfg.model.coefs.l1),
        ('λfix', cfg.model.coefs.debug),
        ('s', cfg.seed),
        ('ep', cfg.epochs),
        ('wep', cfg.warm_epochs),
        ('l', cfg.last_layer_iterations),
    ]

    basename = '__'.join([name + '_' + str(value) if name else str(value)
                          for name, value in fields])
    return basename


def validate_configs(cfg: ExperimentConfig):
    assert cfg.model.base_architecture in model.base_architecture_to_features.keys()

    if cfg.debug.loss:
        if not (len(cfg.debug.classes) or len(cfg.debug.protos)):
            log.warning('Warning!: no classes or prototypes are specified!')

    if cfg.debug.loss == 'iaiabl':
        assert 0 <= cfg.debug.fine_annotation

    if cfg.debug.loss is not None:
        assert cfg.model.coefs.debug > 0

    if not cfg.cpu and not torch.cuda.is_available():
        log.error('cuda not available. Run on CPU by using cpu=True')

    assert cfg.model.gamma > 0


def to_save(target_accuracy, curr_accu, epoch, total_epochs) -> bool:
    return epoch == total_epochs - 1
    # return curr_accu > target_accuracy or epoch == settings.num_train_epochs - 1


def _print_stat(i, log, stat_train: Optional[dict], stat_test: dict, epoch: bool,
                enable_wandb: bool) -> None:
    if enable_wandb:
        if stat_train is not None:
            prefix = 'train_'
            dtrain = {prefix + key: stat_train[key] for key in stat_train.keys() if
                      key not in ['weights', 'all_target', 'all_predicted']}
        else:
            dtrain = {}

        if stat_test is not None:
            prefix = 'test_'
            dtest = {prefix + key: stat_test[key] for key in stat_test.keys() if
                     key not in ['weights', 'all_target', 'all_predicted']}
        else:
            dtest = {}

        wandb.log({**dtrain, **dtest})

    x = PrettyTable()
    keys = list(stat_test.keys())
    for k_to_remove in ['weights', 'all_target', 'all_predicted',
                        'total_features_layer_grad', 'total_add_on_layer_grad',
                        'total_proto_layer_grad', 'total_last_layer_grad',
                        'all_img_paths', 'proto_norm']:
        keys.remove(k_to_remove)
    x.field_names = ['epoch', 'tr/te'] + list(keys)
    if stat_train is not None:
        x.add_row([str(i) + (' e' if epoch else ' i'), 'TR'] + list(
            [round(v, 2) for k, v in stat_train.items() if k in keys]))
    if stat_test is not None:
        x.add_row([str(i) + (' e' if epoch else ' i'), 'TE'] + list(
            [round(v, 2) for k, v in stat_test.items() if k in keys]))
    log.info('\n' + x.get_string())


def get_checkpoint_dict(e: int, ppnet, generator, optimizers: dict):
    return {'epoch': e,
            'rng_state': generator.get_state(),
            'model_state_dict': ppnet.state_dict(),
            'warm_optimizer': optimizers['warm_opt'].state_dict(),
            'joint_optimizer': optimizers['joint_opt'].state_dict(),
            'joint_lr_scheduler': optimizers['joint_lr_scheduler'].state_dict(),
            'last_layer_optimizer': optimizers['last_layer_opt'].state_dict()}


def _bookkeeping_code(model_dir, cfg) -> None:
    base_architecture_type = re.match('^[a-z]*', cfg.model.base_architecture).group(0)
    shutil.copy(src=os.path.join(get_original_cwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(get_original_cwd(), 'settings.py'), dst=model_dir)
    shutil.copy(src=os.path.join(get_original_cwd(), 'features',
                                 base_architecture_type + '_features.py'),
                dst=model_dir)
    shutil.copy(src=os.path.join(get_original_cwd(), 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(get_original_cwd(), 'train_and_test.py'),
                dst=model_dir)
    shutil.copy(src=os.path.join(get_original_cwd(), 'data', 'data_loader.py'),
                dst=model_dir)
    if 'cub' in cfg.data.name:
        shutil.copy(
            src=os.path.join(get_original_cwd(), 'datasets', 'cub200_cropped.dvc'),
            dst=model_dir)

    if cfg.debug.loss == 'aggregation':
        # confounds
        loss_dir = os.path.join(get_original_cwd(), cfg.data.data_path,
                                cfg.data.forbidden_protos_directory)
        dest_copy_dir = os.path.join(model_dir,
                     cfg.data.forbidden_protos_directory)
        if os.path.exists(dest_copy_dir):
            shutil.rmtree(dest_copy_dir)
        shutil.copytree(src=loss_dir, dst=dest_copy_dir)

        # remembering
        loss_dir = os.path.join(get_original_cwd(), cfg.data.data_path,
                                cfg.data.remembering_protos_directory)
        dest_loss_dir = os.path.join(model_dir,
                                         cfg.data.remembering_protos_directory)
        if os.path.exists(dest_loss_dir):
            shutil.rmtree(dest_loss_dir)
        if os.path.exists(loss_dir):
            shutil.copytree(src=loss_dir, dst=dest_loss_dir)


def make_or_load_model(cfg: ExperimentConfig, path_to_model: str,
                       device) -> model.PPNet:
    log.info('construct the model')
    prototype_vector_shape = (
                                 cfg.data.num_classes * cfg.model.num_prototypes_per_class,) \
                             + tuple(cfg.model.prototype_shape)
    hard_constraint_image_dir = None
    ppnet = model.construct_PPNet(base_architecture=cfg.model.base_architecture,
                                  pretrained=True,
                                  img_size=cfg.data.img_size,
                                  prototype_shape=prototype_vector_shape,
                                  num_classes=cfg.data.num_classes,
                                  topk_k=cfg.model.topk_k,
                                  prototype_activation_function=cfg.model.prototype_activation_function,
                                  add_on_layers_type=cfg.model.add_on_layers_type,
                                  hard_constraint_image_dir=hard_constraint_image_dir)

    if path_to_model is not None:
        log.info('load model parameters from ' + path_to_model)
        ppnet.load_state_dict(
            torch.load(path_to_model, map_location=device)['model_state_dict'])

    ppnet.eval()
    return ppnet


def _fix_prototypes(cfg_debug: Debug, ppnet: model.PPNet) -> None:
    if len(cfg_debug.protos) or len(cfg_debug.classes):
        log.info('fixing model phase')
        if cfg_debug.loss in ['aggregation'] or cfg_debug.hard_constraint is True:
            for cl in cfg_debug.classes:
                ppnet.add_class_with_confounder(cl)
            for p in cfg_debug.protos:
                ppnet.re_initialize_prototype_by_id(p)
        elif cfg_debug.loss == 'iaiabl':
            for p in cfg_debug.protos:
                ppnet.re_initialize_prototype_by_id(p)
        else:
            assert len(cfg_debug.protos) == len(cfg_debug.classes)
            for p, cl in zip(cfg_debug.protos, cfg_debug.classes):
                if cfg_debug.loss == 'attribute':
                    ppnet.add_irrelevant_prototype(p, cl)
                elif cfg_debug.loss in ['prototypes', 'kernel']:
                    assert len(cfg_debug.classes) == len(cfg_debug.protos)
                    ppnet.add_irrelevant_concept(p, cl)
                else:
                    raise ValueError(cfg_debug.loss)


def configure_optimizers(ppnet, cfg_model: ModelConfig, load_optimizers: bool,
                         path_to_optimizers: str, device) -> dict:
    joint_optimizer_specs = \
        [{'params': ppnet.features.parameters(),
          'lr': cfg_model.joint_optimizer_lrs['features'],
          'weight_decay': 1e-3},  # bias are now also being regularized
         {'params': ppnet.add_on_layers.parameters(),
          'lr': cfg_model.joint_optimizer_lrs['add_on_layers'],
          'weight_decay': 1e-3},
         {'params': ppnet.prototype_vectors,
          'lr': cfg_model.joint_optimizer_lrs['prototype_vectors']},
         ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer,
                                                         step_size=cfg_model.joint_lr_step_size,
                                                         gamma=cfg_model.gamma,
                                                         verbose=False)

    warm_optimizer_specs = \
        [{'params': ppnet.add_on_layers.parameters(),
          'lr': cfg_model.warm_optimizer_lrs['add_on_layers'],
          'weight_decay': 1e-3},
         {'params': ppnet.prototype_vectors,
          'lr': cfg_model.warm_optimizer_lrs['prototype_vectors']},
         ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(warm_optimizer,
                                                         step_size=4,
                                                         gamma=0.1,
                                                         verbose=False)

    last_layer_optimizer_specs = [
        {'params': ppnet.last_layer.parameters(),
         'lr': cfg_model.last_layer_optimizer_lr}]

    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    if load_optimizers:
        if path_to_optimizers is None:
            raise ValueError('Path to optimizers is None')
        opts = torch.load(path_to_optimizers)
        warm_optimizer.load_state_dict(opts['warm_optimizer'])
        joint_optimizer.load_state_dict(opts['joint_optimizer'])
        joint_lr_scheduler.load_state_dict(opts['joint_lr_scheduler'])
        last_layer_optimizer.load_state_dict(opts['last_layer_optimizer'])

    optimizers = dict(joint_opt=joint_optimizer,
                      joint_lr_scheduler=joint_lr_scheduler,
                      warm_opt=warm_optimizer,
                      warm_lr_scheduler=warm_lr_scheduler,
                      last_layer_opt=last_layer_optimizer)
    return optimizers


def experiment(cfg: ExperimentConfig, generator) -> str:
    base_model_dir = os.getcwd()
    _bookkeeping_code(base_model_dir, cfg)

    device = torch.device("cpu" if cfg.cpu else "cuda")
    log.info(f"Using {device} device")

    data_module = DATASETS[cfg.data.name](cfg, generator, False, device)
    data_module.prepare_data(loss_image=(cfg.debug.loss == 'aggregation'))
    dataset = data_module.get_dataset()

    if cfg.debug.path_to_model is None:
        path_to_model = None
        if cfg.debug.auto_path_to_model:
            prev_round = int(cfg.experiment_name.split('_')[-1]) - 1
            base_experiment_name = '_'.join(cfg.experiment_name.split('_')[:-1])
            if prev_round >= 0:
                path_to_model = glob.glob(os.path.join(get_original_cwd(),
                                                       f'saved_models/{cfg.data.name}/{base_experiment_name}_{prev_round}*/*nopush*.pth.tar'))
                assert len(path_to_model) == 1
                path_to_model = path_to_model[0]
                cfg.debug.path_to_model = path_to_model
    else:
        path_to_model = os.path.join(get_original_cwd(), cfg.debug.path_to_model)

    _debug(path_to_model, dataset, cfg, device, base_model_dir, generator)

    return base_model_dir


def _debug(path_to_model: str, dataset, cfg: ExperimentConfig, device, model_dir: str, generator):
    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log.info(f'  training set size: {len(dataset.train_loader.dataset)}')
    log.info(f'  push set size:     {len(dataset.train_push_loader.dataset)}')
    log.info(f'  test set size:     {len(dataset.test_loader.dataset)}')

    ppnet = make_or_load_model(cfg, path_to_model, device)
    optimizers = configure_optimizers(ppnet, cfg.model, cfg.debug.load_optimizer,
                                      path_to_model, device)
    lrs = [optimizers['joint_lr_scheduler'].get_last_lr()]

    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)

    _fix_prototypes(cfg.debug, ppnet)

    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)

    # if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)

    stat_test = tnt.test(model=ppnet_multi, dataset=dataset,
                         log=log,
                         coefs=cfg.model.coefs, debug_cfg=cfg.debug, device=device)
    _print_stat('debug', log, None, stat_test, epoch=True, enable_wandb=cfg.wandb)

    log.info('start training the model')
    target_accuracy = 0.30
    all_stat_train = []
    all_stat_test = []

    t = tqdm(total=cfg.epochs + len(cfg.push_epochs) * cfg.last_layer_iterations,
             token=os.getenv("TELEGRAM_TOKEN"),
             chat_id=os.getenv('TELEGRAM_CHAT_ID'))
    t.set_description(cfg.experiment_name)
    for epoch in range(cfg.epochs):
        log.info(f'epoch: \t{epoch} / {cfg.epochs}')
        # Stage 1: SGD of layers before the last
        if epoch < cfg.warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            stat_train = tnt.train(model=ppnet_multi,
                                   dataset=dataset,
                                   optimizer=optimizers['warm_opt'],
                                   coefs=cfg.model.coefs,
                                   log=log,
                                   debug_cfg=cfg.debug,
                                   device=device,
                                   dry_run=cfg.dry_run)
            if cfg.debug.loss == 'aggregation' and cfg.data.name in [settings.CovidDatasetConfig.name, settings.Cub200ArtificialConfound.name]:
                optimizers['warm_lr_scheduler'].step()
                lrs.append(optimizers['warm_lr_scheduler'].get_last_lr())
        else:
            tnt.joint(model=ppnet_multi, log=log)
            stat_train = tnt.train(model=ppnet_multi,
                                   dataset=dataset,
                                   optimizer=optimizers['joint_opt'],
                                   coefs=cfg.model.coefs,
                                   debug_cfg=cfg.debug,
                                   device=device, log=log,
                                   dry_run=cfg.dry_run)

            optimizers['joint_lr_scheduler'].step()
            lrs.append(optimizers['joint_lr_scheduler'].get_last_lr())

        stat_test = tnt.test(model=ppnet_multi, dataset=dataset,
                             coefs=cfg.model.coefs, debug_cfg=cfg.debug,
                             device=device, log=log, dry_run=cfg.dry_run)

        save_model_w_condition(state=get_checkpoint_dict(epoch, ppnet, generator, optimizers),
                               model=ppnet,
                               save_path=os.path.join(model_dir,
                                                      f'{epoch}nopush{stat_test["accu"]:.4f}.pth'),
                               to_save=to_save(target_accuracy, stat_test['accu'],
                                               epoch, cfg.epochs),
                               log_wandb=cfg.wandb)
        _print_stat(epoch, log, stat_train, stat_test, epoch=True,
                    enable_wandb=cfg.wandb)
        all_stat_train.append(stat_train)
        all_stat_test.append(stat_test)
        # Stage 2: projection of prototypes
        if epoch >= cfg.push_start and epoch in cfg.push_epochs:
            push.push_prototypes(
                dataset.train_push_loader,
                # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi,
                # pytorch network with prototype_vectors
                class_specific=True,
                preprocess_input_function=preprocess_input_function,
                # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,
                # if not None, prototypes will be saved here
                epoch_number=epoch,
                # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix='prototype-img',
                prototype_self_act_filename_prefix='prototype-self-act',
                proto_bound_boxes_filename_prefix='bb',
                save_prototype_class_identity=True,
                log=log.info,
                device=device
            )

            stat_test = tnt.test(model=ppnet_multi, dataset=dataset,
                                 coefs=cfg.model.coefs, debug_cfg=cfg.debug,
                                 device=device, log=log, dry_run=cfg.dry_run)

            save_model_w_condition(state=get_checkpoint_dict(epoch, ppnet, generator, optimizers),
                                   model=ppnet,
                                   save_path=os.path.join(model_dir,
                                                          f'{epoch}push{stat_test["accu"]:.4f}.pth'),
                                   to_save=True, #to_save(target_accuracy, stat_test['accu'],
                                                  #epoch, cfg.epochs),
                                   log_wandb=cfg.wandb)

            # Stage 3: convex optimization of last layer
            if cfg.model.prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                for it in range(cfg.last_layer_iterations):
                    log.info(f'iteration: \t{it}')
                    stat_train = tnt.train(model=ppnet_multi, dataset=dataset,
                                           optimizer=optimizers['last_layer_opt'],
                                           coefs=cfg.model.coefs,
                                           log=log, debug_cfg=cfg.debug,
                                           device=device, dry_run=cfg.dry_run)
                    stat_test = tnt.test(model=ppnet_multi, dataset=dataset,
                                         log=log, coefs=cfg.model.coefs,
                                         debug_cfg=cfg.debug, device=device, dry_run=cfg.dry_run)

                    save_model_w_condition(state=get_checkpoint_dict(epoch, ppnet, generator, optimizers),
                                           model=ppnet,
                                           save_path=os.path.join(model_dir,
                                                                  f'{epoch}_{it}push{stat_test["accu"]:.4f}.pth'),
                                           to_save=to_save(target_accuracy,
                                                           stat_test['accu'], it,
                                                           cfg.last_layer_iterations),
                                           log_wandb=cfg.wandb)

                    _print_stat(it, log, stat_train, stat_test, epoch=False,
                                enable_wandb=cfg.wandb)
                    all_stat_train.append(stat_train)
                    all_stat_test.append(stat_test)
                    t.update()  # last layer iterations

        t.update()  # epochs
        dump(os.path.join(model_dir, 'stat.pickle'),
             {'cfg': OmegaConf.to_container(cfg),
              'device': device,
              'train': all_stat_train,
              'test': all_stat_test,
              'learning_rate_by_epochs': lrs,
              'classes': None, #dataset.train_push_loader.dataset.class_to_idx,
              'proto_identity': ppnet.prototype_class_identity.detach().numpy()}
             )
        if cfg.dry_run:
            break
        # end epoch

    t.close()


@hydra.main(config_path="conf", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    validate_configs(cfg)
    if cfg.dry_run:
        cfg.wandb = False
        cfg.experiment_name = 'debug_'+cfg.experiment_name
    if cfg.push_epochs is None or len(cfg.push_epochs) == 0:
        cfg.push_epochs = [cfg.epochs - 1]

    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpuid[0]
    log.info(f'Hostname: "{os.uname().nodename}"\n'
             f'Python version: {sys.version}\n'
             f'Start time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n'
             f'Cuda visible devices: {os.environ["CUDA_VISIBLE_DEVICES"]}\n'
             f'Command: {" ".join(sys.argv)}')

    # ----- Reproducibility ----------------
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if not cfg.debug.loss == 'iaiabl':
        # it doesn't work with iaiabl loss (raise an error on torch.nn.Upsample)
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    # for reproducibility purposes
    # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    # -----------------------------------

    start = time.time()
    msg = "{}," + \
          f'{cfg.experiment_name}, {cfg.data.name}, {os.uname().nodename},\n' + \
          'Run time [m]: {}'
    try:
        configuration = omegaconf.OmegaConf.to_container(cfg, resolve=True,
                                                         throw_on_missing=True)
        with open(os.path.join(os.getcwd(), 'config.json'), 'w') as fp:
            json.dump(configuration, fp, indent=4)

        with wandb.init(project='gbm',
                        config=configuration,
                        name=cfg.experiment_name,
                        dir=get_original_cwd(),
                        mode="online" if cfg.wandb else "disabled") as run:

            experiment_dir = experiment(cfg, generator)

        msg = msg.format('Done',
                         int((time.time() - start) / 60)) + f',\ndir: {experiment_dir}'
        log.info(experiment_dir)
        if not cfg.dry_run:
            send_telegram_notification(msg)
    except (Exception, TerminationError) as e:
        if not cfg.dry_run:
            send_telegram_notification(msg.format('Error', int((time.time() - start) / 60)))
        raise e
    log.info(f'Run time [m]: {int((time.time() - start) / 60)}')


if __name__ == '__main__':
    cs = ConfigStore.instance()
    cs.store(name="myconfig", node=ExperimentConfig)
    for dataset_name, config_class in DATASET_CONFIGS.items():
        cs.store(group="data", name=dataset_name, node=config_class)

    main()
