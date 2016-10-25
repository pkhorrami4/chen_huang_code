import numpy


class LandmarkSelector(object):
    def __init__(self, which_feat=[]):
        if which_feat == []:
            self.which_feat = ['head', 'eyebrow_r', 'eyebrow_l', 'nose',
                               'eye_r', 'eye_l', 'mouth']
        else:
            self.which_feat = which_feat

        self.feat_ind_dict = {'head': numpy.arange(34),
                              'eyebrow_r': numpy.arange(34, 44),
                              'eyebrow_l': numpy.arange(44, 54),
                              'nose': numpy.arange(54, 72),
                              'eye_r': numpy.arange(72, 84),
                              'eye_l': numpy.arange(84, 96),
                              'mouth': numpy.arange(96, 136)}

    def run(self, x_batch):
        inds_to_take = []
        for feat in self.feat_ind_dict.keys():
            # print feat
            if feat in self.which_feat:
                inds_to_take.append(self.feat_ind_dict[feat])

        inds_to_take = sorted(numpy.hstack(inds_to_take))
        # print 'inds to take: ', inds_to_take

        return x_batch[:, :, inds_to_take]
