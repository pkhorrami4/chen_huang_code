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
        if fold_type == 'subj_dep':
            assert which_fold in range(3), 'which_fold is out of range'
        else:
            assert which_fold in range(20), 'which_fold is out of range'

    def load(self):
        self.X = numpy.load(
            os.path.join(self.dataset_path, self.feat_type + '.npy'))
        self.y = numpy.load(os.path.join(self.dataset_path, 'y.npy'))
        print 'X:', self.X.shape
        print 'y:', self.y.shape

        fold_inds = numpy.load(
            os.path.join(self.dataset_path, 'folds', self.fold_type,
                         'fold_inds.npy'))
        # for train_inds, test_inds in fold_inds:
        #     print train_inds.shape, test_inds.shape

        train_inds, test_inds = fold_inds[self.which_fold]
        print 'Selected fold:', train_inds.shape, test_inds.shape

        X_train = self.X[train_inds, :]
        y_train = self.y[:, train_inds]
        X_test = self.X[test_inds, :]
        y_test = self.y[:, test_inds]
        print 'X_train:', X_train.shape, 'y_train:', y_train.shape
        print 'X_test:', X_test.shape, 'y_test:', y_test.shape

        data_dict = {}
        data_dict['train'] = {}
        data_dict['test'] = {}
        data_dict['train']['X'] = X_train
        data_dict['train']['y'] = y_train
        data_dict['test']['X'] = X_test
        data_dict['test']['y'] = y_test

        return data_dict
