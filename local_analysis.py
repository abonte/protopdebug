##### MODEL AND DATA LOADING
import argparse
import copy
import logging
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from omegaconf.omegaconf import OmegaConf
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from torch.autograd import Variable

import settings
import train_and_test as tnt
from data.data_loader import DATASETS
from find_nearest import compute_heatmap
from helpers import makedir, find_high_activation_crop, load, create_logger, dump, \
    imsave_with_bbox
from main import make_or_load_model
from preprocess import undo_preprocess_input_function


def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index + 1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1, 2, 0])

    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img


def save_prototype(fname, epoch, index, load_img_dir):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-' + str(epoch),
                                    'prototype-img' + str(index) + '.png'))
    plt.imsave(fname, p_img)


def save_prototype_self_activation(fname, epoch, index, load_img_dir):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-' + str(epoch),
                                    'prototype-img-original_with_self_act' + str(
                                        index) + '.png'))
    plt.imsave(fname, p_img)


def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end,
                                          load_img_dir,
                                          color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-' + str(epoch),
                                        'prototype-img-original' + str(index) + '.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start),
                  (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(fname, p_img_rgb)


def main(test_image_path, test_image_label, load_model_dir, load_model_name,
         ppnet_multi, img_tensor):
    save_analysis_path = os.path.join(load_model_dir, load_model_name.split('.pth')[0],
                                      str(test_image_label),
                                      os.path.basename(test_image_path))
    makedir(save_analysis_path)

    log, logclose = create_logger(
        log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number = int(epoch_number_str)

    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    ##### SANITY CHECK
    # confirm prototype class identity
    load_img_dir = os.path.join(load_model_dir, 'img')

    prototype_info = np.load(os.path.join(load_img_dir, f'epoch-{epoch_number_str}',
                                          f'bb{epoch_number_str}.npy'))
    prototype_img_identity = prototype_info[:, -1]

    log(f'Prototypes are chosen from {len(set(prototype_img_identity))} number of classes.')

    log('Their class identities are: ' + str(prototype_img_identity))

    # confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(
            prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
        log('All prototypes connect most strongly to their respective classes.')
    else:
        log('WARNING: Not all prototypes connect most strongly to their respective classes.')

    # load the test image and forward it through the network
    # normalize = transforms.Normalize(mean=mean, std=std)
    # preprocess = transforms.Compose([
    #     transforms.Resize((img_size, img_size)),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    #
    # img_pil = Image.open(test_image_path)
    # img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    images_test = img_variable.to(device)
    labels_test = torch.tensor([test_image_label])

    logits, min_distances, _ = ppnet_multi(images_test)
    conv_output, distances = ppnet.push_forward(images_test)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    tables = []
    for i in range(logits.size(0)):
        tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
        log(str(i) + ' ' + str(tables[-1]))

    idx = 0
    predicted_cls = tables[idx][0]
    correct_cls = tables[idx][1]
    log('Predicted: ' + str(predicted_cls))
    log('Actual: ' + str(correct_cls))
    original_img = save_preprocessed_img(
        os.path.join(save_analysis_path, 'original_img.png'),
        images_test, idx)

    ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
    makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

    log('Most activated 10 prototypes of this image:')
    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
    class_identity = []
    activation_value = []
    last_layer_weight = []
    for i in range(1, 5):
        log(f'top {i} activated prototype for this image:')
        class_identity.append(prototype_img_identity[sorted_indices_act[-i].item()])
        activation_value.append(array_act[-i].detach().cpu().numpy())

        last_layer_weight.append(
            ppnet.last_layer.weight[predicted_cls][
                sorted_indices_act[-i].item()].item())

        save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                    'top-%d_activated_prototype.png' % i),
                       start_epoch_number, sorted_indices_act[-i].item(), load_img_dir)
        save_prototype_original_img_with_bbox(
            fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                               'top-%d_activated_prototype_in_original_pimg.png' % i),
            epoch=start_epoch_number,
            index=sorted_indices_act[-i].item(),
            bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
            bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
            bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
            bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
            color=(0, 255, 255), load_img_dir=load_img_dir)
        save_prototype_self_activation(
            os.path.join(save_analysis_path, 'most_activated_prototypes',
                         'top-%d_activated_prototype_self_act.png' % i),
            start_epoch_number, sorted_indices_act[-i].item(), load_img_dir)
        log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
        log('prototype class identity: {0}'.format(
            prototype_img_identity[sorted_indices_act[-i].item()]))
        if prototype_max_connection[sorted_indices_act[-i].item()] != \
                prototype_img_identity[sorted_indices_act[-i].item()]:
            log('prototype connection identity: {0}'.format(
                prototype_max_connection[sorted_indices_act[-i].item()]))
        log('activation value (similarity score): {0}'.format(array_act[-i]))
        log('last layer connection with predicted class: {0}'.format(
            ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))

        activation_pattern = prototype_activation_patterns[idx][
            sorted_indices_act[-i].item()].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern,
                                                  dsize=(img_size, img_size),
                                                  interpolation=cv2.INTER_CUBIC)

        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[
                         high_act_patch_indices[0]:high_act_patch_indices[1],
                         high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        # plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                   high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(
            fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                               'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
            img_rgb=original_img,
            bbox_height_start=high_act_patch_indices[0],
            bbox_height_end=high_act_patch_indices[1],
            bbox_width_start=high_act_patch_indices[2],
            bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

        # show the image overlayed with prototype activation map
        heatmap = compute_heatmap(upsampled_activation_pattern)

        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'prototype_activation_map_by_top-%d_prototype.png' % i),
                   overlayed_img)
        log('--------------------------------------------------------------')

    dump(os.path.join(save_analysis_path, 'most_activated_prototypes', 'stats.pickle'),
         {'class_identity': np.array(class_identity),
          'activation_value': activation_value,
          'last_layer_weight': np.array(last_layer_weight),
          'predicted': predicted_cls,
          'true': correct_cls
          })

    ##### PROTOTYPES FROM TOP-k CLASSES
    k = 2
    log('Prototypes from top-%d classes:' % k)
    topk_logits, topk_classes = torch.topk(logits[idx], k=k)
    for i, c in enumerate(topk_classes.detach().cpu().numpy()):
        makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1)))

        log('top %d predicted class: %d' % (i + 1, c))
        log('logit of the class: %f' % topk_logits[i])
        class_prototype_indices = \
            np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        class_prototype_activations = prototype_activations[idx][
            class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        prototype_cnt = 1
        for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            prototype_index = class_prototype_indices[j]
            save_prototype(
                os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                             'top-%d_activated_prototype.png' % prototype_cnt),
                start_epoch_number, prototype_index, load_img_dir)
            save_prototype_original_img_with_bbox(
                fname=os.path.join(save_analysis_path,
                                   'top-%d_class_prototypes' % (i + 1),
                                   'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                epoch=start_epoch_number,
                index=prototype_index,
                bbox_height_start=prototype_info[prototype_index][1],
                bbox_height_end=prototype_info[prototype_index][2],
                bbox_width_start=prototype_info[prototype_index][3],
                bbox_width_end=prototype_info[prototype_index][4],
                color=(0, 255, 255), load_img_dir=load_img_dir)
            save_prototype_self_activation(
                os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                             'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                start_epoch_number, prototype_index, load_img_dir)
            log('prototype index: {0}'.format(prototype_index))
            log('prototype class identity: {0}'.format(
                prototype_img_identity[prototype_index]))
            if prototype_max_connection[prototype_index] != prototype_img_identity[
                prototype_index]:
                log('prototype connection identity: {0}'.format(
                    prototype_max_connection[prototype_index]))
            log('activation value (similarity score): {0}'.format(
                prototype_activations[idx][prototype_index]))
            log('last layer connection: {0}'.format(
                ppnet.last_layer.weight[c][prototype_index]))

            activation_pattern = prototype_activation_patterns[idx][
                prototype_index].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern,
                                                      dsize=(img_size, img_size),
                                                      interpolation=cv2.INTER_CUBIC)

            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(
                upsampled_activation_pattern)
            high_act_patch = original_img[
                             high_act_patch_indices[0]:high_act_patch_indices[1],
                             high_act_patch_indices[2]:high_act_patch_indices[3], :]
            log('most highly activated patch of the chosen image by this prototype:')
            # plt.axis('off')
            plt.imsave(
                os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                             'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                high_act_patch)
            log('most highly activated patch by this prototype shown in the original image:')
            imsave_with_bbox(
                fname=os.path.join(save_analysis_path,
                                   'top-%d_class_prototypes' % (i + 1),
                                   'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                img_rgb=original_img,
                bbox_height_start=high_act_patch_indices[0],
                bbox_height_end=high_act_patch_indices[1],
                bbox_width_start=high_act_patch_indices[2],
                bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

            heatmap = compute_heatmap(upsampled_activation_pattern)
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            log('prototype activation map of the chosen image:')
            # plt.axis('off')
            plt.imsave(
                os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                             'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                overlayed_img)
            log('--------------------------------------------------------------')
            prototype_cnt += 1
        log('***************************************************************')

    if predicted_cls == correct_cls:
        log('Prediction is correct.')
    else:
        log('Prediction is wrong.')

    logclose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', type=str, default='0', help='GPU device ID(s) to used')
    parser.add_argument('model', type=str, help='path to the saved model to analyze')
    args = parser.parse_args()

    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    load_model_dir = os.path.dirname(args.model)
    load_model_path = args.model

    cfg: settings.ExperimentConfig = \
        OmegaConf.create(load(os.path.join(load_model_dir, 'stat.pickle'))['cfg'])

    print('load model from ' + load_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ppnet = make_or_load_model(cfg, load_model_path, device)
    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    data_module = DATASETS[cfg.data.name](cfg, generator, False, device)
    data_module.prepare_data(loss_image=(cfg.debug.loss == 'aggregation'))
    dataset = data_module.get_dataset()
    print('test set size: {0}'.format(len(dataset.test_loader.dataset)))

    # load the test data and check test accuracy
    check_test_accu = False
    if check_test_accu:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        stat_test = tnt.test(model=ppnet_multi, dataset=dataset,
                             log=log,
                             coefs=cfg.model.coefs, debug_cfg=cfg.debug,
                             device=device)
        ConfusionMatrixDisplay.from_predictions(stat_test['all_target'],
                                                stat_test['all_predicted']).plot()
        plt.savefig(os.path.join(os.path.dirname(load_model_path), 'test_all.pdf'))

    count = 0
    for i, data in enumerate(dataset.test_loader):
        for p, t, img in zip(data[2], data[1], data[0]):
            print(p, t)
            if t.item() == 0 and count > 100:
                continue
            count += 1 if t.item() == 0 else 0
            main(p, t.item(), load_model_dir, os.path.basename(load_model_path),
                 ppnet_multi, img)
