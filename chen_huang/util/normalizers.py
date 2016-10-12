import os
import numpy


class LandmarkFeatureNormalizerOld(object):
    def __init__(self, npy_file_path, fold_type, fold):
        assert fold_type in ['subj_dep', 'subj_ind'
                             ], 'fold_type must be subj_dep or subj_ind'

        if fold_type == 'subj_dep':
            assert fold in range(3), 'fold %d does not exist' % fold
        else:
            assert fold in range(20), 'fold %d does not exist' % fold

        self.npy_file_path = npy_file_path
        self.fold_type = fold_type
        self.fold = fold

        self.npy_filename = os.path.join(npy_file_path,
                                         'global_stats_' + fold_type + '.npy')

        global_stats_dict = numpy.load(self.npy_filename)
        global_stats_dict = global_stats_dict[()]

        self.mean_vector = global_stats_dict[fold]['mean']
        self.std_vector = global_stats_dict[fold]['std']

    def run(self, x_batch):
        x_batch_norm = (x_batch - self.mean_vector[None, None, :])
        x_batch_norm /= (self.std_vector[None, None, :] + 1e-6)
        return x_batch_norm


class LandmarkFeatureNormalizer(object):
    def __init__(self, X):
        self.X = X

        self.mean_vector = numpy.mean(self.X, axis=0)
        self.std_vector = numpy.std(self.X, axis=0)

    def run(self, x_batch):
        x_batch_norm = (x_batch - self.mean_vector[None, None, :])
        x_batch_norm /= (self.std_vector[None, None, :] + 1e-6)
        return x_batch_norm
