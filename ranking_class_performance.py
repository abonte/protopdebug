import os

from plot_stat import load_experiments, _get_f1_per_class
import numpy as np

dataset_path = 'datasets/cub200_cropped/clean_top_20'

path_upper = "saved_models/cub200_clean_top20/237_upperbound__vgg16__cub200_clean_top20__e=45__we=5__λfix=0.0__+experiment=natural_upper"
path_lower = "saved_models/cub200_clean_top20/237_lowerbound__vgg16__cub200_clean_top20__e=45__we=5__λfix=0.0__+experiment=natural_base_upper_lower"

print('Load...')
trace_upper = load_experiments([path_upper])
trace_lower = load_experiments([path_lower])

assert len(trace_lower) == 1
assert len(trace_upper) == 1

classes = sorted(os.listdir(os.path.join(dataset_path, 'test_cropped_shuffled')))
classes.remove('.DS_Store')

f1_lower = _get_f1_per_class(trace_lower[0][0], 'test')
f1_upper = _get_f1_per_class(trace_upper[0][0], 'test')
assert np.array(list(f1_lower.values())).flatten().shape == np.array(list(f1_upper.values())).flatten().shape
print(f'f1 lower: {np.mean(np.array(list(f1_lower.values())).flatten())} f1 upper {np.mean(np.array(list(f1_upper.values())).flatten())}')

difference = {}

assert len(f1_upper.keys()) == len(f1_lower.keys()) == len(classes)

for cl in f1_upper.keys():
    difference[classes[int(cl)]] = (f1_upper[cl][-1] - f1_lower[cl][-1], int(cl))

sorted_classes = dict(sorted(difference.items(), key=lambda x: x[1][0])[::-1])

for k, v in sorted_classes.items():
    print(k, v[0], 'idx=', v[1])
