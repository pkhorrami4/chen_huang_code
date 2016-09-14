import os
import numpy


class VideoSequenceGenerator(object):
    def __init__(self, X, y, batch_size=8, max_seq_length=500, verbose=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.verbose = verbose
        
        self.subj_ids = self.y[0, :]
        self.emotion_labels = y[-2, :].astype('int')
        clips_int = self.y[1, :].astype('int')
        self.clip_ids = numpy.mod(clips_int, 3)

        self.unique_subj_ids = numpy.unique(self.subj_ids)
        self.unique_clip_ids = numpy.unique(self.clip_ids)        
    
    def next(self, subj_id=None, emotion_label=None, clip_id=None):
        # If subject id, emotion, or clip is not specified, randomly pick one
        if subj_id is None:            
            subj_id = numpy.random.choice(self.unique_subj_ids)
                    
        if emotion_label is None:
            emotion_label = numpy.random.choice(11)            
        
        if clip_id is None:            
            clip_id = numpy.random.choice(self.unique_clip_ids)            

        if self.verbose:
            print 'Possible subjects: %s' % self.unique_subj_ids
            print 'Chosen subject: %s' % subj_id
            print 'Chosen Emotion Label: %d' % emotion_label
            print 'Possible clip indices: %s' % self.unique_clip_ids
            print 'Clip chosen: %d' % clip_id
        
        # Create mask based on subject id, emotion, and clip number
        subj_mask = (self.subj_ids == subj_id)
        emotion_mask = (self.emotion_labels == emotion_label)
        clip_mask = (self.clip_ids == clip_id)
        inds = numpy.where(numpy.logical_and(numpy.logical_and(subj_mask, emotion_mask), 
                                             clip_mask))[0]
        X_ = self.X[inds, :, :, :]
        y_ = numpy.array(emotion_label)
        
        if self.verbose:
            print X_.shape, y_.shape
        
        return X_, y_
        
    def get_batch(self):
        x_batch = numpy.zeros((self.batch_size, self.max_seq_length,
                              self.X.shape[1], self.X.shape[2], self.X.shape[3]))
        y_batch = numpy.zeros(self.batch_size)
        seq_lengths = numpy.zeros(self.batch_size)
        
        for i in range(self.batch_size):            
            if self.verbose:
                print 'Adding sample %d' % i            
            x_, y_ = self.next()
            seq_length = x_.shape[0]
            x_batch[i, 0:seq_length, :, :, :] = x_
            y_batch[i] = y_
            if self.verbose:
                print ''
            
        return x_batch, y_batch
