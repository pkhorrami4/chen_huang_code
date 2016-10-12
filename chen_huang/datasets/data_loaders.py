import os
import numpy


class LandmarkDataLoader(object):
    def __init__(self, dataset_path, feat_type, fold_type, which_fold):
        self.dataset_path = dataset_path
        self.feat_type = feat_type
        self.fold_type = fold_type
        self.which_fold = which_fold

        assert feat_type in ['landmarks', 'landmarks_diff'
                             ], 'feat_type must be landmarks or landmarks_diff'
        assert fold_type in ['subj_dep', 'subj_ind'
                             ], 'fold_type must be subj_dep or subj_ind'

    def load(self):
        self.X = numpy.load(
            os.path.join(self.dataset_path, self.feat_type + '.npy'))
        self.y = numpy.load(os.path.join(self.dataset_path, 'y.npy'))
        print 'X:', self.X.shape
        print 'y:', self.y.shape

        fold_inds = numpy.load(
            os.path.join(self.dataset_path, 'folds', self.fold_type,
                         'fold_inds.npy'))

        X_all = []
        y_all = []
        for fold in self.which_fold:
            _, inds = fold_inds[fold]
            X_all.append(self.X[inds, :])
            y_all.append(self.y[:, inds])

        X_all = numpy.vstack(X_all)
        y_all = numpy.hstack(y_all)
        print 'X has shape: ', X_all.shape, 'y has shape: ', y_all.shape

        data_dict = {}
        data_dict['X'] = X_all
        data_dict['y'] = y_all

        return data_dict
