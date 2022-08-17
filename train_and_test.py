import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import confusion_matrix, \
    precision_recall_fscore_support
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import settings
from helpers import list_of_distances
from model import PPNet


def _debug_loss(model,
                debug_cfg: settings.Debug,
                loss_loader,
                distances,
                data,
                device):
    fixing_proto_cost = torch.tensor(0.0)
    fixing_other = torch.tensor(0.0)
    if debug_cfg.loss == 'attribute':
        fixing_proto_cost = _weight_loss(model)
    elif debug_cfg.loss == 'aggregation':
        fixing_proto_cost, fixing_other = _forbidding_loss(model, loss_loader,
                                                           device,
                                                           debug_cfg.act_place,
                                                           debug_cfg.class_specific_penalization,
                                                           debug_cfg.epsilon)
    elif debug_cfg.loss == 'kernel':
        fixing_proto_cost = _kernel_loss(model, loss_loader, device)
    elif debug_cfg.loss == 'prototypes':
        fixing_proto_cost = _prototype_loss(model, device)
    elif debug_cfg.loss == 'iaiabl':
        fixing_proto_cost = _iaiabl_loss(model, data, distances, device)
    elif debug_cfg.loss is None:
        pass
    else:
        raise ValueError(debug_cfg.loss)
    return fixing_proto_cost, fixing_other


def _forbidding_loss(model: PPNet, loss_images_loader: DataLoader, device,
                     act_place: str,
                     class_specific_penalization: bool, epsilon: float):
    conf_images, labels = loss_images_loader # next(iter(loss_images_loader))
    distances = model.module.prototype_distances(conf_images)

    if act_place == 'center':
        # center distance value
        dist = distances[:, :, distances.shape[2] // 2, distances.shape[3] // 2]
    elif act_place == 'all':
        # min distance across the image
        dist = -F.max_pool2d(-distances,
                             kernel_size=(distances.size()[2],
                                          distances.size()[3])).view(-1,
                                                                     model.module.num_prototypes)
    else:
        raise ValueError(act_place)

    sim = model.module.distance_2_similarity(dist, epsilon=epsilon)
    # shape [n_confound_images, n_prototypes]
    assert sim.shape == (distances.size()[0], distances.size()[1])

    if class_specific_penalization:
        # class specific penalization
        for cl in model.module.irrelevant_prototypes_class:
            if cl not in labels:
                raise ValueError(
                    f'No confound image for class {cl} in the loss directory,'
                    f'available classes {labels}')

        prototypes_of_confound_class = torch.t(
            model.module.prototype_class_identity[:, labels])
    else:
        # penalize all prototypes
        prototypes_of_confound_class = torch.ones_like(sim, device=device)

    prototypes_of_confound_class = prototypes_of_confound_class.to(device)
    l_dbg = torch.mean(torch.sum(sim * prototypes_of_confound_class, dim=1))
    l_dbg_other = torch.mean(torch.sum(sim * (1 - prototypes_of_confound_class), dim=1))

    return l_dbg, l_dbg_other


def _remembering_loss_exp(model: PPNet, positive_loss_images_loader: DataLoader, act_place: str,
                          device):
    pos_images, labels = positive_loss_images_loader # next(iter(positive_loss_images_loader))

    distances = model.module.prototype_distances(pos_images)

    # shape [n_confound_images, n_prototypes]
    if act_place == 'center':
        # center distance value
        dist = distances[:, :, distances.shape[2] // 2, distances.shape[3] // 2]
    elif act_place == 'all':
        # min distance across the image
        dist = -F.max_pool2d(-distances,
                             kernel_size=(distances.size()[2],
                                          distances.size()[3])).view(-1,
                                                                     model.module.num_prototypes)
    else:
        raise ValueError(act_place)

    sim = model.module.distance_2_similarity(dist)
    prototypes_to_remember = torch.t(
        model.module.prototype_class_identity[:, labels]).to(device)
    l_dbg = torch.mean(torch.sum(sim * prototypes_to_remember, dim=1))

    return l_dbg


def _remembering_loss_quadratic(model: PPNet, positive_loss_images_loader: DataLoader,
                                device):
    pos_images, labels = next(iter(positive_loss_images_loader))
    distances = model.module.prototype_distances(pos_images.to(device))

    # shape [n_confound_images, n_prototypes]
    proto_center_dist = distances[:, :, distances.shape[2] // 2,
                        distances.shape[3] // 2]

    max_dist = (model.module.prototype_shape[1]
                * model.module.prototype_shape[2]
                * model.module.prototype_shape[3])

    prototypes_of_correct_class = torch.t(
        model.module.prototype_class_identity[:, labels]).to(device)
    inverted_distances, _ = torch.max(
        (max_dist - proto_center_dist) * prototypes_of_correct_class, dim=1)
    cluster_cost = torch.mean(max_dist - inverted_distances)

    return cluster_cost


def _prototype_loss(model: PPNet, device):
    l_agg = torch.tensor(0.0).to(device)
    squared_weights_per_class = torch.square(model.module.last_layer.weight)
    n_prototypes = model.module.last_layer.in_features

    for c_i in range(model.module.irrelevant_prototypes_vector.shape[0]):
        proto_class = model.module.irrelevant_prototypes_class[c_i]
        p_i = model.module.irrelevant_prototypes_vector[c_i, ...]
        for c_j in range(n_prototypes):
            p_j = model.module.prototype_vectors[c_j, ...]
            l_agg += torch.sum(p_i * p_j) * squared_weights_per_class[proto_class, c_j]
    return l_agg


def _weight_loss(model: PPNet):
    weights_of_irrelevant_prototypes = model.module.attribution_map * model.module.last_layer.weight
    return torch.sum(torch.square(weights_of_irrelevant_prototypes))


def _iaiabl_loss(model: PPNet, data, distances, device):
    # keep only the images with fine annotation
    with_fa = data[4]  # list of bool
    image = data[0][with_fa]
    label = data[1][with_fa]
    distances = distances[with_fa]
    fine_annotation = data[3][with_fa].to(device)
    fine_annotation = 1 - fine_annotation

    activation = model.module.distance_2_similarity(distances)
    # upsampled_activation shape: [n_images, n_protos, img_size, img_size]
    upsampled_activation = torch.nn.Upsample(
        size=(image.shape[2], image.shape[3]),
        mode='bilinear', align_corners=False)(activation)

    fine_annotation_cost = torch.tensor(0., dtype=torch.float64).to(device)
    # fine_annotation_cost_v2 = torch.tensor(0., dtype=torch.float64).to(device)
    proto_num_per_class = model.module.num_prototypes // model.module.num_classes
    all_white_mask = torch.ones(image.shape[2], image.shape[3]).to(device)
    # fine-annotation mask mi, such that mi takes the value 0 at those pixels
    # that are marked as relevant, and takes the value 1 at other pixels.

    # loss as in repository
    for index in range(image.shape[0]):
        # a = torch.norm(upsampled_activation[index, :label[index] * proto_num_per_class] * (1 * all_white_mask)) + \
        #     torch.norm(upsampled_activation[index, label[index] * proto_num_per_class: (label[index] + 1) * proto_num_per_class] * (1 * fine_annotation[index])) + \
        #     torch.norm(upsampled_activation[index, (label[index] + 1) * proto_num_per_class:] * (1 * all_white_mask))

        b = torch.norm(upsampled_activation[index, :label[index] * proto_num_per_class]) + \
            torch.norm(upsampled_activation[index, label[index] * proto_num_per_class: (label[index] + 1) * proto_num_per_class] * (1 * fine_annotation[index])) + \
            torch.norm(upsampled_activation[index, (label[index] + 1) * proto_num_per_class:])

        fine_annotation_cost += b
    return fine_annotation_cost / image.shape[0]

        # protos_of_correct_class = model.module.prototype_class_identity[:, label[index]]
        # a = torch.norm(
        #     upsampled_activation[index, np.where(protos_of_correct_class == 1)[0]] *
        #     fine_annotation[index])
        # b = torch.norm(upsampled_activation[
        #                    index, np.where(protos_of_correct_class == 0)[
        #                        0]] * all_white_mask)
        # fine_annotation_cost_v2 += a + b

    # protos_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).to(device)
    # protos_of_wrong_classes = 1 - protos_of_correct_class
    # all_white_mask = torch.ones(image.shape[2], image.shape[3]).to(device)
    # a = torch.sum(torch.norm(upsampled_activation * fine_annotation[:,None,::], dim=(2, 3)) * protos_of_correct_class)
    # b = torch.sum(torch.norm(upsampled_activation * all_white_mask[None,...], dim=(2, 3)) * protos_of_wrong_classes)
    # assert torch.isclose(torch.tensor((a + b), dtype=torch.float64), fine_annotation_cost_v2)

    # loss as in the paper
    # protos_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).to(device)
    # protos_of_wrong_classes = 1 - protos_of_correct_class
    # a = torch.sum(torch.norm(upsampled_activation * fine_annotation[:,None,::], dim=(2, 3)) * protos_of_correct_class)
    # b = torch.sum(torch.norm(activation, dim=(2, 3)) * protos_of_wrong_classes)
    # return a + b


def _kernel_loss(model: PPNet, loss_images_loader: DataLoader, device):
    def _kernel(c, c_j):
        tmp = torch.sum(c * c_j, dim=(-1, 1))
        tmp = torch.pow(tmp, 1)
        return torch.mean(tmp)

    def _kernel_normalization(c, c_j):
        return _kernel(c, c_j) / (torch.sqrt(_kernel(c, c) * _kernel(c_j, c_j)))

    ktest_features, _ = next(iter(loss_images_loader))
    current_concept_dist_map = model.module.prototype_distances(ktest_features)

    irrelevant_concept_dist_map = model.module.irrelevant_concept_distances(
        ktest_features)
    squared_last_layer_weights = torch.square(model.module.last_layer.weight)

    l_agg = torch.tensor(.0).to(device)
    # dist (batch_size, n_proto, act_height, act_width)

    proto_class = model.module.irrelevant_prototypes_class[0]
    prototype_banned_class = torch.flatten(torch.nonzero(
        model.module.prototype_class_identity[:, proto_class]))

    ci_dist_maps = irrelevant_concept_dist_map[:, 0, :, :]
    ci_sim_maps = model.module.distance_2_similarity(ci_dist_maps)

    for c_j in prototype_banned_class:
        cj_dist_maps = current_concept_dist_map[:, c_j, :, :]
        cj_sim_maps = model.module.distance_2_similarity(cj_dist_maps)

        l_agg += torch.sum(_kernel_normalization(ci_sim_maps, cj_sim_maps))
        # squared_last_layer_weights[proto_class, c_j])

    return l_agg


# ========================================================
# ========================================================
# ========================================================

def compute_avg_separation_cost(min_distances, prototypes_of_wrong_class):
    avg_separation_cost = \
        torch.sum(min_distances * prototypes_of_wrong_class,
                  dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
    avg_separation_cost = torch.mean(avg_separation_cost)
    return avg_separation_cost


def compute_cluster_separation_cost(max_dist, min_distances, prototypes):
    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes, dim=1)
    cost = torch.mean(max_dist - inverted_distances)
    return cost


# ========================================================


def _compute_total_loss(model, output, data,
                        min_distances, distances, coefs,
                        debug_cfg,
                        loss_loader,
                        positive_loss_images_loader,
                        use_l1_mask, total_loss_component,
                        device, class_weights=None):
    label = data[1]
    cross_entropy = torch.nn.functional.cross_entropy(output, label.to(device),
                                                      weight=class_weights)

    max_dist = (model.module.prototype_shape[1]
                * model.module.prototype_shape[2]
                * model.module.prototype_shape[3])

    # prototypes_of_correct_class [batch_size, num_prototypes]
    prototypes_of_correct_class = torch.t(
        model.module.prototype_class_identity[:, label]).to(device)
    cluster_cost = compute_cluster_separation_cost(max_dist, min_distances,
                                                   prototypes_of_correct_class)

    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
    separation_cost = compute_cluster_separation_cost(max_dist, min_distances,
                                                      prototypes_of_wrong_class)

    # calculate avg cluster cost
    avg_separation_cost = compute_avg_separation_cost(min_distances,
                                                      prototypes_of_wrong_class)

    fixing_proto_cost, fixing_other = _debug_loss(model,
                                                  debug_cfg,
                                                  loss_loader,
                                                  distances,
                                                  data,
                                                  device)

    if debug_cfg.loss == 'aggregation' and positive_loss_images_loader is not None:
        remembering_loss = _remembering_loss_exp(model, positive_loss_images_loader,
                                                 debug_cfg.act_place,
                                                 device)
    else:
        remembering_loss = torch.tensor(0.)

    if use_l1_mask:
        l1_mask = 1 - torch.t(model.module.prototype_class_identity).to(
            device)
        l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
    else:
        l1 = model.module.last_layer.weight.norm(p=1)

    total_loss_component['crs_ent'] += coefs.crs_ent * cross_entropy.item()
    total_loss_component['clst'] += coefs.clst * cluster_cost.item()
    total_loss_component['sep'] += coefs.sep * separation_cost.item()
    total_loss_component['avg_sep'] += avg_separation_cost.item()
    total_loss_component['agg'] += coefs.debug * fixing_proto_cost.item()
    total_loss_component['agg_other'] += coefs.debug * fixing_other.item()
    total_loss_component['remembering'] -= coefs.rem * remembering_loss.item()

    loss = (coefs.crs_ent * cross_entropy
            + coefs.clst * cluster_cost
            + coefs.sep * separation_cost
            + coefs.l1 * l1
            + coefs.debug * fixing_proto_cost
            - coefs.rem * remembering_loss)
    return loss


def _train_or_test(model: PPNet, dataloader, device,
                   optimizer: Optional[Optimizer] = None,
                   use_l1_mask: bool = True,
                   coefs: Optional[settings.ModelConfig.Lambdas] = None,
                   debug_cfg=None,
                   positive_loss_loader=None,
                   log=print,
                   loss_loader=None,
                   dry_run: bool = False,
                   class_weights=None):
    """
    model: the multi-gpu model
    optimizer: if None, will be test evaluation
    """
    is_train = optimizer is not None
    start = time.time()
    total_loss_component = dict(crs_ent=0, clst=0, sep=0, avg_sep=0, agg=0, agg_other=0,
                                remembering=0)
    total_grad = dict(features=0, add_on=0, proto_layer=0, last_layer=0)
    total_loss, n_batches = 0, 0
    all_predicted, all_target = [], []
    all_img_paths = []  # for reproducibility check

    progress_bar = tqdm(total=len(dataloader))
    progress_bar.set_description('train' if is_train else 'test')

    # print(random.random())
    # print(np.random.random())
    # print(torch.randint(0,1000, (1,)))

    for i, data in enumerate(dataloader):
        input_data = data[0] #.to(device)
        if is_train:
            # all_img_paths.extend(data[2])
            all_img_paths.extend([])
            # print('aaa', data[2])
            # print('bbb', torch.linalg.norm(data[0], dim=(1,2)))
            # print('bbb', np.linalg.norm(data[0].detach().numpy().astype(np.float64)))

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        with grad_req:
            # nn.Module has implemented __call__() function so no need to call .forward
            output, min_distances, distances = model(input_data)

            if is_train:
                loss = _compute_total_loss(model, output, data,
                                           min_distances, distances, coefs,
                                           debug_cfg,
                                           loss_loader,
                                           positive_loss_loader,
                                           use_l1_mask, total_loss_component,
                                           device, class_weights)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            all_predicted.extend(predicted.tolist())
            all_target.extend(data[1].tolist())
            # pr, rc, f1, _ = precision_recall_fscore_support(data[1].tolist(), predicted.detach().tolist(),
            #                                                 average='macro')
            # print('abc ',pr, rc, f1)
            # if n_batches == 1:
            #     break

            n_batches += 1

        # compute gradient and do SGD step
        if is_train:
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_grad = _get_grad_values(model, total_grad)

        del data, output, predicted, min_distances
        progress_bar.update()
        if dry_run:
            break

    end = time.time()
    progress_bar.close()

    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))

    log.info('\n' + np.array2string(confusion_matrix(all_target, all_predicted)))
    log.info('\n' + classification_report(all_target, all_predicted))

    pr, rc, f1, _ = precision_recall_fscore_support(all_target, all_predicted,
                                                    average='macro')

    statistic = {
        'time': end - start,
        'loss': total_loss / n_batches,
        # 'grad': total_features_layer_grad + total_add_on_layer_grad + total_proto_layer_grad + total_last_layer_grad,
        'total_features_layer_grad': total_grad['features'] / n_batches,
        'total_add_on_layer_grad': total_grad['add_on'] / n_batches,
        'total_proto_layer_grad': total_grad['proto_layer'] / n_batches,
        'total_last_layer_grad': total_grad['last_layer'] / n_batches,
        # 'norm_proto_layer': model.module.prototype_vectors.cpu().clone().detach().norm(p=1).item(),
        'cross_ent': total_loss_component['crs_ent'] / n_batches if is_train else 0,
        'cluster': total_loss_component['clst'] / n_batches if is_train else 0,
        'separation': total_loss_component['sep'] / n_batches if is_train else 0,
        'avg_sep': total_loss_component['avg_sep'] / n_batches if is_train else 0,
        'forgetting_loss': (total_loss_component['agg'] / n_batches) if is_train else 0,
        'remembering_loss': (
                total_loss_component['remembering'] / n_batches) if is_train else 0,
        'fix_other': total_loss_component['agg_other'] / n_batches if is_train else 0,
        'l1': model.module.last_layer.weight.cpu().clone().detach().norm(p=1).item(),
        'dist_pair': p_avg_pair_dist.item(),
        'weights': model.module.last_layer.weight.cpu().clone().detach().numpy(),
        'accu': accuracy_score(all_target, all_predicted),
        'all_target': all_target,
        'all_predicted': all_predicted,
        'all_img_paths': all_img_paths,
        'proto_norm': [proto.cpu().clone().detach().norm(p=1).item() for proto in
                       model.module.prototype_vectors] if is_train else [],
        'f1': f1,
        'rc': rc,
        'pr': pr
    }

    return statistic


def _get_grad_values(model: PPNet, total_grad: dict) -> dict:
    with torch.no_grad():
        for i in model.module.features.features.modules():
            if isinstance(i, nn.Conv2d) and i.weight.grad is not None:
                total_grad['features'] += torch.norm(i.weight.grad).item()
                total_grad['features'] += torch.norm(i.bias.grad).item()

        for i in model.module.add_on_layers.modules():
            if isinstance(i, nn.Conv2d) and i.weight.grad is not None:
                total_grad['add_on'] += torch.norm(i.weight.grad).item()
                total_grad['add_on'] += torch.norm(i.bias.grad).item()

        total_grad['proto_layer'] += torch.norm(
            model.module.prototype_vectors.grad).item()

        if model.module.last_layer.weight.grad is not None:
            total_grad['last_layer'] += torch.norm(
                model.module.last_layer.weight.grad).item()
    return total_grad


def train(model, dataset, device, optimizer,
          debug_cfg: settings.Debug,
          coefs: Optional[settings.ModelConfig.Lambdas] = None,
          log=print, dry_run: bool = False):
    assert (optimizer is not None)

    log.info('\ttrain')
    model.train()
    return _train_or_test(model=model,
                          dataloader=dataset.train_loader,
                          optimizer=optimizer,
                          coefs=coefs,
                          debug_cfg=debug_cfg,
                          log=log,
                          loss_loader=dataset.loss_loader,
                          positive_loss_loader=dataset.positive_loss_loader,
                          device=device,
                          dry_run=dry_run,
                          class_weights=dataset.class_weights)


def test(model, dataset, device, debug_cfg, log=print, coefs=None, dry_run=False):
    log.info('\ttest')
    model.eval()
    return _train_or_test(model=model,
                          dataloader=dataset.test_loader,
                          optimizer=None,
                          debug_cfg=debug_cfg,
                          log=log,
                          loss_loader=dataset.loss_loader,
                          positive_loss_loader=None,
                          coefs=coefs,
                          device=device,
                          dry_run=dry_run)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log.info('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log.info('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log.info('\tjoint')
