import argparse
import os
import re
from collections import defaultdict
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from omegaconf.omegaconf import OmegaConf
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from torch.utils.data.dataloader import DataLoader
import train_and_test as tnt

import find_nearest
import settings
from data.data_loader import DATASETS
from helpers import makedir, load
from main import make_or_load_model, _print_stat
from model import PPNet
from preprocess import preprocess_input_function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def save_prototype_original_img_with_bbox(load_img_dir, fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end,
                                          color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-' + str(epoch),
                                        'prototype-img-original' + str(index) + '.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start),
                  (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    # plt.imshow(p_img_rgb)
    # plt.axis('off')
    plt.imsave(fname, p_img_rgb)


def activation_precision(load_model_dir: str,
                         model: PPNet,
                         data_set: DataLoader,
                         epoch_number_str: int,
                         preprocess_input_function=None,
                         percentile: int = 95,
                         per_proto: bool = False):
    """Interpretability metric (IAIA-BL paper)"""

    print('Compute activation precision')
    n_prototypes = model.module.num_prototypes

    precisions = []
    per_proto_hp = defaultdict(list)

    for idx, data in enumerate(data_set):
        print('\tbatch {}'.format(idx))
        if True:
            with_fine_annotation = data[4]
            search_batch_input = data[0][with_fine_annotation]
            search_y = data[1][with_fine_annotation]
            fine_anno = data[3][with_fine_annotation]
            if len(search_y) == 0:
                print(f'Skip {idx}')
                continue
        else:
            search_batch_input = data[0]
            search_y = data[1]
            fine_anno = data[3]

        if preprocess_input_function is not None:
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input)
        else:
            search_batch = search_batch_input

        with torch.no_grad():
            search_batch = search_batch.to(device)
            fine_anno = fine_anno.to(device)
            protoL_input_torch, proto_dist_torch = model.module.push_forward(
                search_batch)

        proto_acts = model.module.distance_2_similarity(proto_dist_torch)

        proto_acts = torch.nn.Upsample(
            size=(search_batch.shape[2], search_batch.shape[3]), mode='bilinear',
            align_corners=False)(proto_acts)

        # confirm prototype class identity
        load_img_dir = os.path.join(load_model_dir, 'img')

        prototype_info = np.load(os.path.join(load_img_dir,
                                              f'epoch-{epoch_number_str}',
                                              f'bb{epoch_number_str}.npy'))
        prototype_img_identity = prototype_info[:, -1]
        print('Prototypes are chosen from ' + str(
            len(set(prototype_img_identity))) + ' number of classes.')
        print('Their class identities are: ' + str(prototype_img_identity))

        proto_acts_ = np.copy(proto_acts.detach().cpu().numpy())
        fine_anno_ = np.copy(fine_anno.detach().cpu().numpy())
        assert proto_acts_.shape[0] == fine_anno_.shape[0]

        for img_idx, (activation_maps_per_proto, fine_annotation) in enumerate(
                zip(proto_acts_, fine_anno_)):
            # for every test img
            for j in range(n_prototypes):
                # for each proto
                if prototype_img_identity[j] == search_y[img_idx]:
                    # if proto class matches img class

                    activation_map = activation_maps_per_proto[j]
                    threshold = np.percentile(activation_map, percentile)
                    mask = np.ones(activation_map.shape)
                    mask[activation_map < threshold] = 0
                    mask = mask * activation_map
                    assert fine_annotation.shape == mask.shape
                    denom = np.sum(mask)
                    num = np.sum(mask * fine_annotation)
                    pr = num / denom

                    precisions.append(pr)
                    per_proto_hp[j].append(pr)

    if per_proto:
        per_proto_hp_list = []
        for k, v in per_proto_hp.items():
            per_proto_hp_list.append((k, v))
        per_proto_hp_list.sort(key=lambda x: x[0])
        return per_proto_hp_list
    else:
        return np.average(np.asarray(precisions))


def main(load_model_path: str):
    load_model_dir = os.path.dirname(load_model_path)
    model_name = os.path.basename(load_model_path)
    epoch_iter = re.search(r'\d+(_\d+)?', model_name).group(0)
    start_epoch_number = re.search(r'\d+', epoch_iter).group(0)
    run_cfg: settings.ExperimentConfig = \
        OmegaConf.create(load(os.path.join(load_model_dir, 'stat.pickle'))['cfg'])
    run_cfg.data.data_path = run_cfg.data.relative_data_path

    print('load model from ' + load_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ppnet = make_or_load_model(run_cfg, load_model_path, device)
    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)
    ppnet_multi.eval()

    load_img_dir = os.path.join(load_model_dir, 'img')
    prototype_info = np.load(os.path.join(load_img_dir, f'epoch-{start_epoch_number}',
                                          f'bb{start_epoch_number}.npy'))

    k = 50

    data_module = DATASETS[run_cfg.data.name](run_cfg, None, global_analysis=False)
    data_module.prepare_data(loss_image=True)
    dataset = data_module.get_dataset()

    stat_test = tnt.test(model=ppnet_multi, dataset=dataset,
                         log=log,
                         coefs=run_cfg.model.coefs, debug_cfg=run_cfg.debug,
                         device=device)
    _print_stat('debug', log, None, stat_test, epoch=True, enable_wandb=False)
    ConfusionMatrixDisplay.from_predictions(stat_test['all_target'],
                                            stat_test['all_predicted']).plot()
    plt.savefig(os.path.join(os.path.dirname(load_model_path),
                             f'{os.path.basename(load_model_path).split(".pth")[0]}_test_all.pdf'))
    del dataset

    data_module = DATASETS[run_cfg.data.name](run_cfg, None, global_analysis=True)
    data_module.prepare_data(True)
    dataset = data_module.get_dataset()

    # pr = activation_precision(load_model_dir,
    #                           ppnet_multi, dataset.test_loader, start_epoch_number,
    #                           preprocess_input_function=preprocess_input_function,
    #                           per_proto=True)
    # dump(os.path.join(load_model_dir,
    #                   f'{epoch_iter}prototypes_activation_precision_testset.pickle'),
    #      pr)
    # print(pr)

    for data, suffix in [
                        (dataset.loss_loader, '_nearest_kernel_set'),
                        (dataset.train_push_loader, '_nearest_train'),
                        # (dataset.test_loader, '_nearest_test'),
                         # (dataset.kernel_set_loader, '_nearest_kernel_set')
                         ]:

        root_dir_for_saving_images = load_model_path.split('.pth')[0] + suffix
        makedir(root_dir_for_saving_images)

        find_nearest.find_k_nearest_patches_to_prototypes(
            dataloader=data,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi,
            # pytorch network with prototype_vectors
            k=k,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            full_save=True,
            root_dir_for_saving_images=root_dir_for_saving_images,
            log=print)

        # save prototypes in original images
        for j in range(ppnet.num_prototypes):
            makedir(os.path.join(root_dir_for_saving_images, str(j)))
            save_prototype_original_img_with_bbox(
                load_img_dir=load_img_dir,
                fname=os.path.join(root_dir_for_saving_images, str(j),
                                   'prototype_in_original_pimg.png'),
                epoch=start_epoch_number,
                index=j,
                bbox_height_start=prototype_info[j][1],
                bbox_height_end=prototype_info[j][2],
                bbox_width_start=prototype_info[j][3],
                bbox_width_end=prototype_info[j][4],
                color=(0, 255, 255))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0')
    parser.add_argument('model', type=str, help='path to the saved model to analyze')
    args = parser.parse_args()

    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    main(args.model)
