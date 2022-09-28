import glob
import os
import random
import shutil
import string

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from omegaconf.omegaconf import OmegaConf
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture

import settings
from find_nearest import compute_heatmap
from helpers import load, makedir
from plot_stat import load_experiments


def silhouette_analysis(X: npt.ArrayLike, range_n_clusters: list,
                        original_img_size: int, file_basename: str,
                        dest_path: str
                        ) -> GaussianMixture:
    # scikit-learn
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    silh_avgs = []
    gms = []
    fig, axs = plt.subplots(len(range_n_clusters), 2)
    fig.set_size_inches(18, 17)
    for row, n_clusters in enumerate(range_n_clusters):
        clusterer = GaussianMixture(n_components=n_clusters, random_state=seed,
                max_iter=150, tol=1e-6, n_init=5).fit(X)
        #print(f'GM stats, converged={clusterer.converged_} n_iter_={clusterer.n_iter_}')
        cluster_labels = clusterer.predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        silh_avgs.append(silhouette_avg)
        gms.append(clusterer)

        # Create a subplot with 1 row and 2 columns
        ax1, ax2 = axs[row, 0], axs[row, 1]

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title(
            f"The silhouette plot for the various clusters. n_clusters = {n_clusters}")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], original_img_size - X[:, 1], marker=".", s=30, lw=0, alpha=0.7,
            c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.means_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            original_img_size - centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], original_img_size - c[1], marker="$%d$" % i, alpha=1,
                        s=50, edgecolor="k")

        ax2.set_title(
            f"The visualization of the clustered data. n_clusters = {n_clusters}")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

    plt.savefig(os.path.join(dest_path, f'silhouette_{file_basename}.png'))
    print(f'best n of clusters {gms[np.argmax(silh_avgs)].means_.shape[0]}')
    return gms[np.argmax(silh_avgs)]


def cut_and_save_patches(gm: GaussianMixture,
                         X: npt.ArrayLike,
                         original_img: npt.ArrayLike,
                         upsampled_act_img: npt.ArrayLike,
                         mask: npt.ArrayLike,
                         file_basename: str,
                         dest_path: str):
    cluster_labels = gm.predict(X)
    n_components = gm.means_.shape[0]

    # # order cluster by center activation
    # cluster_center_act = dict()
    # for idx in range(n_components):
    #     center_x, center_y = gm.means_[idx, 0], gm.means_[idx, 1]
    #     cluster_center_act[str(idx)] = upsampled_act_img[center_y, center_x]
    #
    # dict(sorted(cluster_center_act.items(), key=lambda item: item[1]))

    for i, letter in zip(range(n_components), string.ascii_lowercase):
        points_in_cluster = X[cluster_labels == i]
        lower_y: int = np.amin(points_in_cluster[:, 1])
        upper_y: int = np.amax(points_in_cluster[:, 1])
        lower_x: int = np.amin(points_in_cluster[:, 0])
        upper_x: int = np.amax(points_in_cluster[:, 0])

        _figure_for_user_experiment(dest_path,
                                    file_basename,
                                    letter,
                                    lower_x, upper_x, lower_y, upper_y,
                                    original_img,
                                    upsampled_act_img,
                                    mask)

def cut_and_save_patches_contours(gm: GaussianMixture,
                         X: npt.ArrayLike,
                         original_img: npt.ArrayLike,
                         upsampled_act_img: npt.ArrayLike,
                         mask: npt.ArrayLike,
                         file_basename: str,
                         dest_path: str):

    img = np.array(mask, np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    for i, letter in zip(range(len(contours)), string.ascii_lowercase):
        upper_x = np.amax([c[0][0] for c in contours[i]])
        lower_x = np.amin([c[0][0] for c in contours[i]])
        upper_y = np.amax([c[0][1] for c in contours[i]])
        lower_y = np.amin([c[0][1] for c in contours[i]])

        print(letter, 'area', (upper_x-lower_x)*(upper_y-lower_y), ' Discarded: ', (upper_x-lower_x)*(upper_y-lower_y) < 200)
        if (upper_x-lower_x)*(upper_y-lower_y) < 200:
            print('discard')
            dp = os.path.join(dest_path, 'discarded')
            makedir(dp)
        else:
            dp = dest_path
        _figure_for_user_experiment(dp,
                                    file_basename,
                                    letter,
                                    lower_x, upper_x, lower_y, upper_y,
                                    original_img,
                                    upsampled_act_img,
                                    mask)

def _figure_for_user_experiment(dest_path: str,
                                file_basename: str,
                                letter: str,
                                lower_x: int, upper_x: int, lower_y: int, upper_y: int,
                                original_img: npt.ArrayLike,
                                upsampled_act_img: npt.ArrayLike,
                                mask_upsampled_act_img: npt.ArrayLike):

    fig, ax1 = plt.subplots(1, 1)
    #ax1.imshow(original_img)
    mask_cluster = np.zeros_like(mask_upsampled_act_img)
    # TODO verificare se upper_y +1
    mask_cluster[lower_y:upper_y + 1, lower_x:upper_x + 1] = 1
    mask_cluster = mask_upsampled_act_img * mask_cluster
    heatmap = compute_heatmap(upsampled_act_img * mask_cluster)
    # heatmap = heatmap + ((1 - mask_cluster[..., np.newaxis]) * 0.4)
    # if overlay and shaded original image
    heatmap = heatmap * mask_cluster[..., np.newaxis]
    ax1.imshow(0.4 * heatmap + 0.6 * (original_img / 255), vmin=0.0, vmax=1.0)

    # if overlay and original images
    # mask_cluster = mask_cluster[..., np.newaxis]
    # heatmap = 0.4 * heatmap * mask_cluster
    # image = 0.6 * original_img * mask_cluster + original_img * (1-mask_cluster)
    # ax1.imshow(heatmap + (image / 255), vmin=0.0, vmax=1.0)

    ax1.axis('off')
    #ax2.axis('off')
    # ax1.text(-200, original_img.shape[0] / 2, 'Image', size='x-large')
    # ax1.text(-200, original_img.shape[0] + 170, 'Patch',
    #          size='x-large')
    # fig.set_dpi(100)
    plt.tight_layout(w_pad=4)
    plt.savefig(os.path.join(dest_path, f'exp_{file_basename}_{letter}.png'), bbox_inches='tight')
    cut = original_img[lower_y:upper_y + 1, lower_x:upper_x + 1]  # on matrix: y and then x
    plt.imsave(
        fname=os.path.join(dest_path, f'{file_basename}_{letter}.png'),
        arr=cut,
        vmin=0.0,
        vmax=1.0)
    plt.close(fig)


def _figure_for_user_experiment_v1(dest_path: str, file_basename: str, letter: str,
                                lower_x: int, upper_x: int, lower_y: int, upper_y: int,
                                original_img: npt.ArrayLike):
    img_bgr_uint8 = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    color = (0, 255, 255)
    cv2.rectangle(img_bgr_uint8,
                  (lower_x, lower_y),
                  (upper_x - 1, upper_y - 1),
                  color,
                  thickness=2)
    # cv2.putText(img_bgr_uint8, str(i),
    #             (lower_x, lower_y-6),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1,color ,2,cv2.LINE_AA)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    # plt.imsave(fname=file_name_template.format('all_cuts'),
    #           arr=img_rgb_float)
    # figure for user experiment
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(img_rgb_float)
    cut = original_img[lower_y:upper_y, lower_x:upper_x]
    ax2.imshow(cut)  # on matrix: y and then x
    ax1.axis('off')
    ax2.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
    spines_color = 'yellow'
    ax2.spines['bottom'].set_color(spines_color)
    ax2.spines['top'].set_color(spines_color)
    ax2.spines['right'].set_color(spines_color)
    ax2.spines['left'].set_color(spines_color)
    ax1.text(-200, img_rgb_float.shape[0] / 2, 'Image', size='x-large')
    ax1.text(-200, img_rgb_float.shape[0] + 170, 'Patch',
             size='x-large')
    fig.set_dpi(200)
    plt.savefig(os.path.join(dest_path, f'exp_{file_basename}_{letter}.png'))
    plt.imsave(
        fname=os.path.join(dest_path, 'cuts', f'{file_basename}_{letter}.png'),
        arr=cut,
        vmin=0.0,
        vmax=1.0)


def is_contiguous_area(mask: npt.ArrayLike) -> bool:
    img = np.array(mask, np.uint8)
    _, contours, = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print('number of contours: ' + str(len(contours[0])))
    return len(contours[0]) == 1


def debug_main():
    act = load('0/nearest-1_act.pickle')
    original_img = cv2.imread('0/nearest-1_original.png')

    original_img_size = 224
    upsampled_act_img_j = cv2.resize(act[0],
                                     dsize=(original_img_size, original_img_size),
                                     interpolation=cv2.INTER_CUBIC)
    percentile = 95
    extract_highest_activated_patches(original_img,
                                      original_img_size,
                                      upsampled_act_img_j, 'name{}.png', percentile)


def extract_highest_activated_patches(original_img: npt.ArrayLike,
                                      original_img_size: int,
                                      upsampled_act_img: npt.ArrayLike,
                                      file_basename: str,
                                      dest_path: str,
                                      i: int,
                                      percentile: int = 95):
    threshold = np.percentile(upsampled_act_img, percentile)
    mask = np.ones(upsampled_act_img.shape)
    mask[upsampled_act_img < threshold] = 0

    fig, ax = plt.subplots()
    ax.imshow(mask)
    fig.savefig(os.path.join(dest_path, f'mask_{file_basename}.png'))

    if is_contiguous_area(mask):
        print('extract only one patch')
        high_act_patch_indices = np.load(
            os.path.join(dest_path, '..', f'nearest-{i}_high_act_patch_indices.npy'))
        lower_y = high_act_patch_indices[0]
        upper_y = high_act_patch_indices[1]
        lower_x = high_act_patch_indices[2]
        upper_x = high_act_patch_indices[3]

        _figure_for_user_experiment(dest_path, file_basename, 'a',
                                    lower_x, upper_x, lower_y, upper_y,
                                    original_img,
                                    upsampled_act_img, mask)

    else:
        print('extract multiple patches')
        X = np.transpose(np.nonzero(mask))
        X[:, [1, 0]] = X[:, [0, 1]]  # swap columns, from yx to xy

        range_n_clusters = [2, 3, 4, 5]
        #clusterer = silhouette_analysis(X, range_n_clusters, original_img_size,
        #                                file_basename, dest_path)
        #cut_and_save_patches(clusterer, X, original_img, upsampled_act_img, mask,
        #                     file_basename, dest_path)

        cut_and_save_patches_contours(None, X, original_img, upsampled_act_img, mask,
                             file_basename, dest_path)


if __name__ == '__main__':
    seed = 624
    np.random.seed(seed)
    random.seed(seed)
    # debug_main()
    # path = 'saved_models/cub200_clean_top20/237_lowerbound__vgg16__cub200_clean_top20__e=45__we=5__λfix=0.0__+experiment=natural_base_upper_lower/29_9push0.6591_nearest_train'
    # path = 'saved_models/cub200_clean_top20/237_1user__vgg16__cub200_clean_top20__e=5__we=5__λfix=100.0__+experiment=natural_aggregation/4_1push0.7663_nearest_train'
    path = 'saved_models/cub200_clean_top20/237_2user__vgg16__cub200_clean_top20__e=5__we=5__λfix=100.0__+experiment=natural_aggregation/4_1push0.7803_nearest_train'
    classes = [0, 8, 14, 6, 15]
    img_per_proto = 5
    user_exp_path = os.path.join(path, 'user_experiment')
    makedir(user_exp_path)
    makedir(os.path.join(user_exp_path, 'form_figures'))
    makedir(os.path.join(user_exp_path, 'cuts'))
    traces = load_experiments([os.path.dirname(path)])
    first_run_cfg: settings.ExperimentConfig = OmegaConf.create(traces[0][0]['cfg'])
    for proto_idx in range(
            first_run_cfg.model.num_prototypes_per_class * first_run_cfg.data.num_classes):
        class_idx = int(int(proto_idx) / first_run_cfg.model.num_prototypes_per_class)
        if classes is not None:
            if class_idx not in classes:
                continue
            dest_path = os.path.join(path, str(proto_idx), 'auto_patch_extraction')
            makedir(dest_path)
            for i in range(1, img_per_proto + 1):
                print('================================')
                print(f'cl={class_idx} pr={proto_idx} i={i}')
                act = load(
                    os.path.join(path, str(proto_idx), f'nearest-{i}_act.pickle'))
                original_img = cv2.imread(
                    os.path.join(path, str(proto_idx), f'nearest-{i}_original.png'))
                original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

                upsampled_act_img = cv2.resize(act[0],
                                               dsize=(
                                                     first_run_cfg.data.img_size,
                                                     first_run_cfg.data.img_size),
                                               interpolation=cv2.INTER_CUBIC)

                file_basename = f'c={class_idx}_p={proto_idx}_i={i}'
                extract_highest_activated_patches(original_img,
                                                  first_run_cfg.data.img_size,
                                                  upsampled_act_img,
                                                  file_basename,
                                                  dest_path,
                                                  i)

            for exp_image in glob.glob(os.path.join(dest_path, f'exp_*')):
                shutil.copy(src=exp_image,
                            dst=os.path.join(user_exp_path, 'form_figures'))
            for exp_image in glob.glob(os.path.join(dest_path, f'c=*')):
                shutil.copy(src=exp_image,
                            dst=os.path.join(user_exp_path, 'cuts'))

    # randomize form figures
    makedir(os.path.join(user_exp_path, 'random_form_figures'))
    img_list = os.listdir(os.path.join(user_exp_path, 'form_figures'))
    img_list.sort()
    img_list = [name for name in img_list if name.startswith('exp_')]
    random_img_list = random.sample(img_list, len(img_list))
    for i, img in enumerate(random_img_list):
        shutil.copy(src=os.path.join(user_exp_path, 'form_figures', img),
                    dst=os.path.join(user_exp_path, 'random_form_figures', f'{str(i).zfill(2)}H{img}'))

    print('Done!')
