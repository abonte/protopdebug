#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import pickle
import shutil
from collections import defaultdict
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines, markers
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from omegaconf.omegaconf import OmegaConf
from sklearn.metrics._classification import classification_report
from sklearn.metrics import f1_score
from prettytable import PrettyTable

import settings
from helpers import makedir
from main import _get_basename

LOSS_LABEL = {
    'attribute': 'attribute',
    'aggregation': 'ProtoPDebug',
    'prototypes': 'prototypes',
    'kernel': 'kernel',
    'iaiabl': 'IAIA-BL',
    'hard_constraint': 'hard const'
}

COLOR = {
    'attribute': 'limegreen',
    'aggregation': 'red',
    'prototypes': 'violet',
    'kernel': 'orange',
    'iaiabl': 'darkorchid',
    'hard_constraint': 'blue'
}

METRICS = {
    'f1': '$F_1$',
    'cross_ent': r'$\lambda_{\mathrm{ce}}\;\ell_{\mathrm{ce}}$'
}

MARKER_IAIABL = {
    0.05: '.',
    0.2: 'v',
    0.1: '<',
    0.15: '*',
    1.0: 'D'
}
cm = plt.cm.get_cmap('tab20c')
IAIABL_COLOR = {
    3.0: cm.colors[3],
    0.05: cm.colors[2],
    0.2: cm.colors[1],
    1.0: cm.colors[0],
}


def get_label_color_marker(cfg: settings.ExperimentConfig):
    if cfg.debug.loss == 'iaiabl':
        if cfg.debug.fine_annotation > 1:
            return f'{LOSS_LABEL[cfg.debug.loss]} n={int(cfg.debug.fine_annotation)}', \
               IAIABL_COLOR[cfg.debug.fine_annotation], \
               '>'
        else:
            return f'{LOSS_LABEL[cfg.debug.loss]} {int(cfg.debug.fine_annotation * 100)}%', \
               IAIABL_COLOR[cfg.debug.fine_annotation], \
               MARKER_IAIABL[cfg.debug.fine_annotation]
    elif cfg.debug.loss is not None:
        return LOSS_LABEL[cfg.debug.loss], COLOR[cfg.debug.loss], None
    elif (not cfg.data.train_on_segmented_image and cfg.data.name not in [
        settings.Cub200Clean5.name]) and cfg.debug.loss is None:
        # lower bound baseline
        return 'ProtoPNet', 'black', '*'
    elif (cfg.data.train_on_segmented_image or cfg.data.name in [
        settings.Cub200Clean5.name]) and cfg.debug.loss is None:
        # upper bound baseline
        return 'ProtoPNet clean', 'green', 'o'
    else:
        raise ValueError()


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def plot_single_experiments(traces, output_path):
    metrics = ['f1', 'f1_per_class',
               'cross_ent', 'forgetting_loss', 'remembering_loss', 'cluster',
               'separation',
               'rc', 'pr', 'accu',
               'weights', 'grad', 'learning_rate_by_epochs', 'proto_norm',
               'conf_matrix']
    fig, axs = plt.subplots(len(metrics), 2, figsize=(15, len(metrics) * 3))

    cfg = OmegaConf.create(traces[0]['cfg'])
    for exp in traces:
        for row, metric in zip(axs, metrics):
            for elem_in_row, state in zip(row, ['train', 'test']):
                if metric == 'grad':
                    for grad_label in ['total_features_layer_grad',
                                       'total_add_on_layer_grad',
                                       'total_proto_layer_grad',
                                       'total_last_layer_grad']:
                        grad_value = np.array(
                            [epoch[grad_label] for epoch in exp[state]])
                        elem_in_row.set_yscale('log')
                        elem_in_row.plot(grad_value, label=grad_label)
                        elem_in_row.legend()
                elif metric == 'weights':
                    x = np.array([epoch[metric] for epoch in exp[state]])
                    # irr_protos = range(5)  # list(exp['args']['irrelevant_prototypes'])
                    # irr_classes = range(5)  # list(exp['args']['irr_proto_classes'])
                    # # x shape (epochs, classes, prototypes)
                    # assert x.shape[1] <= len(lines.lineStyles.keys())
                    # assert x.shape[2] <= len(markers.MarkerStyle.markers.keys())
                    # for c, style in zip(irr_classes, lines.lineStyles.keys()):
                    #     for p, marker in zip(irr_protos,
                    #                          markers.MarkerStyle.markers.keys()):
                    # elem_in_row.plot(x[:, c, p], label=f'class {c} proto {p}',
                    #                  linestyle=style, marker=marker)
                    for c in range(x.shape[1]):
                        for p in range(x.shape[2]):
                            elem_in_row.plot(x[:, c, p], label=f'class {c} proto {p}')
                    # elem_in_row.legend()
                elif metric == 'f1_per_class':
                    x = _get_f1_per_class(exp, state)

                    for label, value in x.items():
                        elem_in_row.plot(value, label=label)
                        elem_in_row.legend()
                    elem_in_row.legend()
                elif metric == 'learning_rate_by_epochs':
                    #if state == 'train':
                    #    elem_in_row.plot(exp[metric])
                    pass
                elif metric == 'proto_norm':
                    if state == 'train':
                        x = np.array([epoch[metric] for epoch in exp[state]])
                        for i in range(x.shape[1]):
                            elem_in_row.plot(x[:, i], label=i)
                        elem_in_row.set_yscale('log')
                        elem_in_row.legend()
                elif metric == 'conf_matrix':
                    ConfusionMatrixDisplay.from_predictions(
                        exp[state][-1]['all_target'],
                        exp[state][-1]['all_predicted'],
                        ax=elem_in_row)
                    elem_in_row.grid(False)
                else:
                    x = np.array([epoch[metric] for epoch in exp[state]])
                    elem_in_row.plot(x)  # , label=LOSS_LABEL[args.loss],
                    # color=COLOR[args.loss])

                elem_in_row.set_title(f'{state} {metric}')
                elem_in_row.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, _get_basename(cfg) + '.pdf'),
                bbox_inches='tight',
                pad_inches=0)


def _get_f1_per_class(exp, state='train'):
    x = defaultdict(lambda: [])
    for epoch in exp[state]:
        output = classification_report(epoch['all_target'],
                                       epoch['all_predicted'],
                                       output_dict=True,
                                       zero_division=0)

        for label, value in output.items():
            if label in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            x[int(label)].append(value['f1-score'])
    return x


def _check_same_data_across_runs(traces, paths):
    for state in ['train', 'test']:
        for repetition in range(len(traces[0])):
            exp_one = [epoch['all_img_paths'] for epoch in traces[0][repetition][state]]
            exp_one_path = paths[0]
            for run, run_second_path in zip(traces[1:], paths[1:]):
                run_second = [epoch['all_img_paths'] for epoch in
                              run[repetition][state]]
                if exp_one != run_second:
                    raise ValueError(
                        f'Run with different data order\n{exp_one_path}\n!=\n{run_second_path}')


def _get_array(experiment: list, metric_name: str, state: str):
    """return  [n repetition, n epochs] """
    assert type(experiment) == list
    y = []
    for run in experiment:
        y.append([epoch[metric_name] for epoch in run[state]])
    return np.atleast_2d(np.array(y))


def plot_multiple_experiments(traces: list, output_path: str):
    for metric_name in ['f1', 'cross_ent']:
        for state in ['train', 'test']:
            if metric_name == 'cross_ent' and state == 'test':
                continue
            fig, axs = plt.subplots(figsize=(7, 4))
            dataset_name = traces[0][0]['cfg'].data.name
            for i, experiment in enumerate(traces):
                if i == 2:
                    # to fix number of items per row in the legend
                    axs.plot([2], [.7], color='w', alpha=0, label=' ')
                run_cfg = OmegaConf.create(experiment[0]['cfg'])
                label, color, marker = get_label_color_marker(run_cfg)
                y = _get_array(experiment, metric_name, state)

                axs.errorbar(x=range(y.shape[1]),
                             y=np.mean(y, axis=0),
                             yerr=np.std(y, axis=0) / np.sqrt(y.shape[0]),
                             label=label,
                             color=color,
                             marker=marker,
                             errorevery=(i%3, 3),
                             markevery=(i%3, 3),
                             )

            axs.legend(ncol=2, fontsize='large')
            # axs.set_title(f'{state} {metric_name}')
            axs.set_xlabel('training epochs', fontsize='xx-large')
            axs.set_ylabel(METRICS[metric_name], fontsize='xx-large')
            axs.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.yticks(fontsize='x-large')
            fig.savefig(os.path.join(output_path,
                                     f'{dataset_name}_{state}_{metric_name}_multiple_stat.pdf'),
                        bbox_inches='tight',
                        pad_inches=0)


def plot_prototypes(modeldirs: list, traces: list, output_path: str, classes: list, method_name: str):
    first_run_cfg: settings.ExperimentConfig = OmegaConf.create(traces[0][0]['cfg'])
    n_cols = len(traces)
    num_classes_to_plot = first_run_cfg.data.num_classes if classes is None else len(
        classes)
    n_rows = num_classes_to_plot * first_run_cfg.model.num_prototypes_per_class
    figsize = (n_cols * 3.4, n_rows * 3.4)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    print(axs.shape)

    assert len(modeldirs) == len(traces)
    for exp_idx, (model_dir, trace) in enumerate(zip(modeldirs, traces)):
        row_count = 0
        assert len(trace) == 1
        dest_proto_folder = os.path.join(os.path.dirname(output_path), f'{method_name}_{exp_idx}')
        makedir(dest_proto_folder)

        base_path = sorted(glob.glob(os.path.join(model_dir, 'img/epoch-*')))[-1]
        for proto_idx in range(
                first_run_cfg.model.num_prototypes_per_class * first_run_cfg.data.num_classes):

            class_idx = int(proto_idx / first_run_cfg.model.num_prototypes_per_class)
            if classes is not None:
                if class_idx not in classes:
                    continue

            path_to_proto = os.path.join(base_path,
                                         f'prototype-img-original_with_self_act{proto_idx}.png')
            act = np.load(os.path.join(base_path, f"prototype-self-act{proto_idx}.npy"))

            if n_cols == 1:
                ax = axs[row_count]
            else:
                ax = axs[row_count, exp_idx]
            ax.set_title(f'class={class_idx} proto_idx={proto_idx}\n'
                         f'max_act={np.max(act):.2f}\n'
                         f'mean_act={np.mean(act):.2f}')
            cmap = plt.cm.get_cmap("jet")
            norm = mpl.colors.Normalize(vmin=np.min(act), vmax=np.max(act))
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8,
                         format='%.2f')
            ax.imshow(plt.imread(path_to_proto))
            ax.axis('off')

            shutil.copy2(src=path_to_proto, dst=dest_proto_folder)

            row_count += 1

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def plot_experiment_iterations_in_same_plot(axs, modeldirs: list, traces: list,
                                            name: str, classes: list, stats: dict):
    ax_idx = 0
    for state in ['train', 'test']:

        run_cfg: settings.ExperimentConfig = OmegaConf.create(traces[0][0]['cfg'])
        label, color, marker = get_label_color_marker(run_cfg)

        # F1
        metric_name = 'f1'
        all_exp_data = list()
        for experiment in traces:
            if classes is None:
                data = _get_array(experiment, metric_name, state)
            else:
                store_runs = []
                for run in experiment:
                    run_f1 = []
                    for epoch in run[state]:
                        run_f1.append(
                            f1_score(epoch['all_target'], epoch['all_predicted'],
                                     average='macro', labels=classes))
                    store_runs.append(run_f1)
                data = np.atleast_2d(np.array(store_runs))

            # data = (n repetition, n epochs)
            all_exp_data.append(data)

        all_exp_data = np.hstack(all_exp_data)
        y = np.mean(all_exp_data, axis=0)
        yerr = np.std(all_exp_data, axis=0) / np.sqrt(all_exp_data.shape[0])

        if state == 'test':
            stats['all'][metric_name] = (y[-1], yerr[-1])

        axs[ax_idx].errorbar(x=range(len(y)),
                             y=y,
                             yerr=yerr,
                             label=label,
                             color=color,
                             marker=marker,
                             errorevery=(10, 3),
                             markevery=(10, 3),
                             )

        axs[ax_idx].set_title(f'{name} {state} {metric_name}')
        # axs[ax_idx].set_xlabel('training epochs')
        axs[ax_idx].set_ylabel(METRICS[metric_name])
        axs[ax_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[ax_idx].set_ylim([0.4, 1.03])
        ax_idx += 1

        # F1 PER CLASS
        y = defaultdict(lambda: [])
        for experiment in traces:
            store_runs = defaultdict(lambda: [])
            for run in experiment:
                f1 = _get_f1_per_class(run, state)
                for cls, v in f1.items():
                    store_runs[cls].append(v)
            for cls, runs_data in store_runs.items():
                y[cls].append(runs_data)

        for cls, v in y.items():
            if classes is not None:
                if cls not in classes:
                    continue
            all_exp_data = np.hstack(v)
            mean = np.mean(all_exp_data, axis=0)
            std = np.std(all_exp_data, axis=0) / np.sqrt(all_exp_data.shape[0])
            axs[ax_idx].errorbar(x=range(len(mean)),
                                 y=mean,
                                 yerr=std,
                                 label=cls,
                                 errorevery=(10, 3))
            if state == 'test':
                stats[cls]['f1'] = (mean[-1], yerr[-1])

        axs[ax_idx].legend(ncol=2)
        axs[ax_idx].set_title(f'{name} {state}  f1 per class')
        # axs[ax_idx].set_xlabel('training epochs')
        axs[ax_idx].set_ylabel(METRICS['f1'])
        axs[ax_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[ax_idx].set_ylim([-0.1, 1.03])
        ax_idx += 1

    # Add vertical lines between iterations
    last_epoch = -1  # start count from 0
    for experiment in traces[0:-1]:
        last_epoch += len(experiment[0][state])
        for i in range(ax_idx):
            axs[i].axvline(x=last_epoch, color='red', linestyle='--')

    # ACTIVATION PRECISION
    for experiment in traces:
        assert len(experiment) == 1
    apr_mean = defaultdict(lambda: [])
    apr_std = defaultdict(lambda: [])
    for mdir in modeldirs:
        path_to_data = glob.glob(
            os.path.join(mdir, '*prototypes_activation_precision_testset.pickle'))
        if len(path_to_data) == 0:
            break
        path_to_data.sort()
        path_to_data = path_to_data[-1]
        obj = load(path_to_data)
        for proto_idx, pr in obj:
            cls_idx = proto_idx // run_cfg.model.num_prototypes_per_class
            if classes is not None:
                if cls_idx not in classes:
                    continue
            apr_mean[proto_idx].append(np.mean(pr))
            apr_std[proto_idx].append(np.std(pr) / np.sqrt(len(pr)))

    for proto_idx in apr_mean.keys():
        if state == 'test':
            cls_idx = proto_idx // run_cfg.model.num_prototypes_per_class
            field_name = f'act_pr_{proto_idx % run_cfg.model.num_prototypes_per_class}'
            stats[cls_idx][field_name] = apr_mean[proto_idx][-1], apr_std[proto_idx][-1]
        axs[ax_idx].errorbar(x=range(len(apr_mean[proto_idx])), y=apr_mean[proto_idx],
                             yerr=apr_std[proto_idx], label=proto_idx)

    axs[ax_idx].legend(ncol=2)
    axs[ax_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[ax_idx].set_title(f'{name} activation precision')
    axs[ax_idx].set_ylim([0.0, 1.03])
    axs[ax_idx].set_xlabel('debugging iterations')
    axs[ax_idx].set_ylabel('apr')
    ax_idx += 1

    # CONF MATRIX, last epoch of last iteration
    axs[ax_idx].set_title('Confusion Matrix on test set')
    ConfusionMatrixDisplay.from_predictions(traces[-1][0]['test'][-1]['all_target'],
                                            traces[-1][0]['test'][-1]['all_predicted'],
                                            ax=axs[ax_idx],
                                            labels=classes,
                                            # xticks_rotation='vertical'
                                            )
    axs[ax_idx].grid(False)
    return stats


def plot_iaiabl_vs_our_method(traces, output_path):
    our_method_f1 = None
    hard_constraints_f1 = None
    iaiabl_f1 = dict()
    for experiment in traces:
        cfg: settings.ExperimentConfig = experiment[0]['cfg']
        y = _get_array(experiment, 'f1', 'test')
        if cfg.debug.loss == 'iaiabl':
            if cfg.debug.fine_annotation in iaiabl_f1.keys():
                raise ValueError(
                    'Multiple run iaia-bl runs with the same fine annotation rate')
            iaiabl_f1[cfg.debug.fine_annotation] = (
                np.mean(y, axis=0), np.std(y, axis=0) / np.sqrt(y.shape[0]))
        elif cfg.debug.loss == 'aggregation' and our_method_f1 is None:
            our_method_f1 = (
                np.mean(y, axis=0), np.std(y, axis=0) / np.sqrt(y.shape[0]))
        elif cfg.debug.hard_constraint is True and hard_constraints_f1 is None:
            hard_constraints_f1 = (
                np.mean(y, axis=0), np.std(y, axis=0) / np.sqrt(y.shape[0]))
        elif cfg.debug.loss == 'aggregation' and our_method_f1 is not None:
            raise ValueError('Multiple run with loss aggregation')
        elif cfg.debug.hard_constraint is True and hard_constraints_f1 is not None:
            raise ValueError('Multiple run with hard constraints')
        else:
            continue

    fig, axs = plt.subplots(figsize=(7, 4))
    x = list(range(len(iaiabl_f1.keys())))
    if our_method_f1 is not None:
        axs.errorbar(x=x,
                     y=[our_method_f1[0][-1] for _ in x],
                     yerr=[our_method_f1[1][-1] for _ in x],
                     color=COLOR['aggregation'],
                     label=LOSS_LABEL['aggregation'])

    if not len(iaiabl_f1.keys()):
        axs.errorbar(x=x,
                     y=[value[0][-1] for (_, value) in sorted(iaiabl_f1.items())],
                     yerr=[value[1][-1] for (_, value) in sorted(iaiabl_f1.items())],
                     color=COLOR['iaiabl'],
                     label=LOSS_LABEL['iaiabl'])

    labels = [float(elem) * 100 for elem in list(sorted(iaiabl_f1.keys()))]
    axs.set_xticks(np.arange(len(iaiabl_f1.keys())), labels)
    axs.set_ylabel(METRICS['f1'])
    axs.set_xlabel('% of fine annotated images')
    plt.legend()
    plt.savefig(os.path.join(output_path, f'{cfg.data.name}_test_f1_iaiabl_vs_our.pdf'),
                bbox_inches='tight',
                pad_inches=0)


def _check_order_of_experiments_is_correct(traces, paths):
    previous_model = None
    for i, t in enumerate(traces):
        print(t[0]['cfg'].experiment_name)
        if previous_model is not None:
            current_used_model = os.path.basename(os.path.dirname(
                t[0]['cfg'].debug.path_to_model))
            if previous_model != current_used_model:
                print('previous: ' + previous_model + '\ncurrent' + current_used_model)
                raise ValueError('order of the directories matter')

        previous_model = os.path.basename(paths[i])


def load_experiments(modeldirs: list) -> list:
    traces = []
    for path in modeldirs:
        path_to_stat = os.path.join(path, 'stat.pickle')
        if os.path.exists(path_to_stat):
            # single repetition experiment
            stat = load(path_to_stat)
            stat['cfg'] = OmegaConf.create(stat['cfg'])
            traces.append([stat])
        else:
            # multiple repetition experiment (artificial confound experiment)
            trace_repetitions = []
            for repetition in os.listdir(path):
                if os.path.isdir(os.path.join(path, repetition)):
                    if not os.path.exists(
                            os.path.join(path, repetition, 'stat.pickle')):
                        print(
                            f'Warning: skip "{os.path.join(path, repetition)}"')
                        continue
                    stat = load(os.path.join(path, repetition, 'stat.pickle'))
                    stat['cfg'] = OmegaConf.create(stat['cfg'])
                    trace_repetitions.append(stat)
            if len(trace_repetitions) == 0:
                print(f'Warning: skip "{path}"')
                continue
            traces.append(trace_repetitions)
            # shape list traces (n_experiments, n_repetition, [train, test], epochs, metric)
    return traces


def main(script_args):
    makedir(script_args.output_path)

    if script_args.experiment_name == 'comparison':
        traces = load_experiments(script_args.modeldir)
        if len(traces) == 1 and len(traces[0]) == 1:
            output_path = script_args.modeldir if script_args.output_path is None else script_args.output_path
            plot_single_experiments(traces[0], output_path)
        else:
            # _check_same_data_across_runs(traces, script_args.modeldir)
            plot_multiple_experiments(traces, script_args.output_path)
            plot_iaiabl_vs_our_method(traces, script_args.output_path)

    elif script_args.experiment_name == 'debugging':
        lines = {'ourloss': script_args.ourloss,
                 'upperbound': script_args.upperbound,
                 'lowerbound': script_args.lowerbound,
                 'iaiabl': script_args.iaiabl,
                 }
        stats = {}
        fig, axs = plt.subplots(6, len(lines.keys()), figsize=(23, 25))
        for (method_name, paths), column in zip(lines.items(), axs.T):
            if paths is None:
                continue

            method_stats = {cl: {} for cl in script_args.classes + ['all']}
            traces = load_experiments(paths)
            dataset_name = traces[0][0]['cfg'].data.name
            # #_check_order_of_experiments_is_correct(traces, paths)
            script_args.classes.sort()
            method_stats = plot_experiment_iterations_in_same_plot(column, paths,
                                                                   traces, method_name,
                                                                   script_args.classes,
                                                                   method_stats)
            stats[method_name] = method_stats

            plot_prototypes(modeldirs=paths,
                            traces=traces,
                            output_path=os.path.join(script_args.output_path,
                                                     f'{method_name}_{dataset_name}_prototypes_by_iteration.png'),
                            method_name=method_name,
                            classes=script_args.classes)

        plot_result_table(lines, stats, script_args.classes)

        plt.tight_layout()
        fig.savefig(os.path.join(script_args.output_path,
                                 f'{dataset_name}_train_test__multiple_iteration.pdf'),
                    bbox_inches='tight', pad_inches=0)

    else:
        raise ValueError()


def plot_result_table(lines, stats, classes):
    return
    all_rows = []
    name_fields = ['cls']
    for cls in classes + ['all']:
        tmp_row = [str(cls)]
        for method in ['ourloss', 'upperbound', 'lowerbound']:
            for metric in ['f1', 'act_pr_0', 'act_pr_1']:
                if len(all_rows) == 0:
                    name_fields.append(f'{method}_{metric}')
                if metric != 'f1' and cls == 'all':
                    tmp_row.append('None')
                else:
                    mean, err = stats[method][cls][metric]
                    tmp_row.append(f'{round(mean, 2)}$\pm${round(err, 2)}')
            tmp_row[-1] += '\n'
        all_rows.append(tmp_row)

    # name_fields = ['cls', 'metric']
    # row_names_added = defaultdict(lambda: False)
    # for method_name, s in stats.items():
    #     row_counter = 0
    #     name_fields.append(method_name)
    #     for cls, metric_values in s.items():
    #         for metric, (mean, err) in metric_values.items():
    #             if not row_names_added[row_counter]:
    #                 all_rows[row_counter] = [str(cls), metric.replace('_', ' ')]
    #                 row_names_added[row_counter] = True
    #             all_rows[row_counter].extend([f'{round(mean, 2)}$\pm${round(err, 2)}'])
    #             row_counter += 1

    print('## Latex')
    print(' & '.join(name_fields), r" \\")
    for row in all_rows:
        print('  & '.join(row), r" \\ ")

    x = PrettyTable()
    x.field_names = name_fields
    x.add_rows(all_rows)
    print(x)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, dest='output_path',
                        help='output folder (default is modeldir)',
                        default=None)
    subparser = parser.add_subparsers(title='Experiments', dest='experiment_name')
    parser_exp1 = subparser.add_parser('comparison', help='exp 1 in paper')
    parser_exp1.add_argument('-modeldir', nargs='+', type=str,
                             help='list of model directories')

    parser_exp2 = subparser.add_parser('debugging',
                                       help='exp 2 in paper, debugging sequence')
    parser_exp2.add_argument('--ourloss', nargs='+', type=str,
                             help='model directories sequence for our loss')
    parser_exp2.add_argument('--upperbound', nargs='+', type=str,
                             help='model directories sequence for upperbound (segmented images)')
    parser_exp2.add_argument('--lowerbound', nargs='+', type=str,
                             help='model directories sequence for lowerbound (no corrections)')
    parser_exp2.add_argument('--iaiabl', nargs='+', type=str)
    parser_exp2.add_argument('--classes', nargs='+', type=int,
                             help='the (0-based) index of the classes used to compute the f1 score')

    script_args = parser.parse_args()

    if script_args.experiment_name == 'debugging':
        if script_args.classes is not None:
            print(f'Warning: metrics computed on this classes {script_args.classes}')
    else:
        script_args.classes = None

    if script_args.classes is not None:
        script_args.classes.sort()
    plt.style.use('ggplot')
    main(script_args)
    print('Done!')
