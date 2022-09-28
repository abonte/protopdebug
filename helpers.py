import os
import pickle
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2,
                     dim=1)


def make_one_hot(target, target_one_hot):
    target = target.view(-1, 1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)


def makedir(path: str):
    """
    if path does not exist in the file system, create it
    """
    if not os.path.exists(path):
        os.makedirs(path)


def find_high_activation_crop(activation_map, percentile: int = 95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


def imsave_with_bbox(fname: str,
                     img_rgb,
                     bbox_height_start: int,
                     bbox_height_end: int,
                     bbox_width_start: int,
                     bbox_width_end: int,
                     color=(0, 255, 255)):

    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start),
                  (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    # plt.imshow(img_rgb_float)
    # plt.axis('off')
    plt.imsave(fname, img_rgb_float)

def save_model_w_condition(state: dict, model, save_path: str, to_save: bool,
                           log_wandb: bool = False):
    """
    model: this is not the multigpu model
    """
    if to_save:
        # torch.save(obj=model, f=save_path)
        torch.save(state, save_path + '.tar')
        # if log_wandb:
        #     artifact = wandb.Artifact('model', type='model')
        #     artifact.add_file(save_path + '.tar')
        #     wandb.log_artifact(artifact)


def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    counter = [0]

    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
        # Question: do we need to flush()

    return logger, f.close


def dump(path, what, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(what, fp, **kwargs)


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)
