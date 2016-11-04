import os
import numpy
import tensorflow as tf


class RecurrentModel(object):
    def __init__(self, model_dict, verbose=False):
        # Load hyperparameter settings
        self._set_hyper_parameters(model_dict, verbose)

        self.inputs = tf.placeholder(
            tf.float32, [None, self.max_sequence_length, self.input_size])
        self.targets = tf.placeholder(tf.int32,
                                      [None, self.max_sequence_length])
        self.sequence_lengths = tf.placeholder(tf.int32, [None])
        self.mask = tf.placeholder(tf.bool, [None, self.max_sequence_length])
        self.batch_size = tf.shape(self.inputs)[0]

        if self.cell_type == 'basic_rnn':
            self.single_rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
        elif self.cell_type == 'basic_lstm':
            self.single_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(
                self.state_size, state_is_tuple=True)
        elif self.cell_type == 'lstm':
            self.single_rnn_cell = tf.nn.rnn_cell.LSTMCell(
                self.state_size, state_is_tuple=True)
        elif self.cell_type == 'gru':
            self.single_rnn_cell = tf.nn.rnn_cell.GRUCell(self.state_size)
        else:
            raise Exception('cell_type incorrectly specified')

        self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
            [self.single_rnn_cell] * self.num_layers, state_is_tuple=True)
        self.outputs, self.states = tf.nn.dynamic_rnn(
            self.rnn_cell,
            self.inputs,
            dtype=tf.float32,
            sequence_length=self.sequence_lengths)
        self.outputs_r = tf.reshape(
            self.outputs,
            [self.batch_size * self.max_sequence_length, self.state_size])

        # Output matrix parameters
        self.softmax_w = tf.get_variable(
            "softmax_w", [self.state_size, self.num_classes],
            initializer=tf.truncated_normal_initializer(
                mean=0.0, stddev=1.0))
        self.softmax_b = tf.get_variable(
            "softmax_b", [self.num_classes],
            initializer=tf.constant_initializer(0.0))
        self.logits = tf.matmul(self.outputs_r,
                                self.softmax_w) + self.softmax_b
        self.softmax = tf.nn.softmax(self.logits)

        if verbose:
            print 'outputs shape: ', self.outputs.get_shape()
            print 'outputs_r shape: ', self.outputs_r.get_shape()
            print 'logits shape: ', self.logits.get_shape()
            print 'softmax shape: ', self.softmax.get_shape()

        # Compute loss
        self.targets_r = tf.reshape(
            self.targets, [self.batch_size * self.max_sequence_length])
        self.loss = tf.nn.seq2seq.sequence_loss_by_example(
            [self.logits], [self.targets_r],
            [tf.ones([self.batch_size * self.max_sequence_length])])
        self.loss_sum = tf.reduce_sum(
            tf.reshape(self.loss, [self.batch_size, self.max_sequence_length])
            * tf.cast(self.mask, tf.float32),
            reduction_indices=1)
        self.total_cost = tf.reduce_mean(self.loss_sum / tf.cast(
            self.sequence_lengths, tf.float32))

        # Compute accuracy
        self.accuracy = self._compute_accuracy(self.softmax, self.targets_r,
                                               self.mask)
        self.accuracy_clip = self._compute_accuracy_clip(
            self.softmax, self.targets, self.mask)
        self.preds_clip = self._compute_preds_clip(self.softmax, self.targets,
                                                   self.mask)

        if verbose:
            print 'targets_r shape: ', self.targets_r.get_shape()
            print 'loss shape: ', self.loss.get_shape()
            print 'loss_sum shape: ', self.loss_sum.get_shape()
            print 'total cost shape: ', self.total_cost.get_shape()

        # Create optimizer
        #lr = tf.Variable(self.learning_rate, trainable=False)
        global_step = tf.Variable(0, trainable=False)
        starter_lr = tf.Variable(self.learning_rate, trainable=False)
        if self.anneal_lr:
            decay_steps = 1400
            decay_rate = 0.1
            lr = tf.train.exponential_decay(
                starter_lr,
                global_step,
                decay_steps,
                decay_rate,
                staircase=True)
        else:
            lr = starter_lr

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.total_cost, tvars), 1)
        # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        self.train_op = self.optimizer.apply_gradients(
            zip(grads, tvars), global_step=global_step)

        # Setup Session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        #self.sess = tf.Session()
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Grab summaries for Tensorboard
        tf.scalar_summary('learning_rate', lr)
        tf.scalar_summary('total_cost', self.total_cost)
        tf.scalar_summary('accuracy', self.accuracy)
        tf.scalar_summary('accuracy_clip', self.accuracy_clip)
        self.merged = tf.merge_all_summaries()
        self.summary_writer_train = tf.train.SummaryWriter(
            self.summary_path + '/train', self.sess.graph)
        self.summary_writer_val = tf.train.SummaryWriter(
            self.summary_path + '/val', self.sess.graph)

        # Add saver
        self.saver = tf.train.Saver(max_to_keep=None)

        self.sess.run(tf.initialize_all_variables())

    def _set_hyper_parameters(self, model_dict, verbose=False):
        self.input_size = model_dict['input_size']
        # self.batch_size = model_dict['batch_size']
        self.max_sequence_length = model_dict['max_sequence_length']
        self.num_classes = model_dict['num_classes']
        self.state_size = model_dict['state_size']
        self.cell_type = model_dict['cell_type']
        self.num_layers = model_dict['num_layers']
        self.learning_rate = model_dict['learning_rate']
        self.anneal_lr = model_dict['anneal_lr']
        self.save_path = model_dict['save_path']
        self.summary_path = model_dict['summary_path']
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        if verbose:
            print 'input_size: %d' % self.input_size
            # print 'batch_size: %d' % self.batch_size
            print 'max_sequence_length: %d' % self.max_sequence_length
            print 'num_classes: %d' % self.num_classes
            print 'state_size: %d' % self.state_size
            print 'cell_type: %s' % self.cell_type

    def _compute_preds_clip(self, probs, targets, mask):
        probs_r = tf.reshape(
            probs,
            [self.batch_size, self.max_sequence_length, self.num_classes])
        mask_expanded = tf.cast(tf.expand_dims(mask, 2), tf.float32)
        mask_repeated = tf.tile(mask_expanded, [1, 1, self.num_classes])
        seq_lengths_expanded = tf.cast(
            tf.expand_dims(self.sequence_lengths, 1), tf.float32)

        avg_probs = tf.reduce_sum(
            probs_r * mask_repeated,
            reduction_indices=1) / seq_lengths_expanded
        class_preds = tf.cast(tf.argmax(avg_probs, 1), tf.int32)

        return class_preds

    def get_preds_clip(self, x_batch, seq_lengths_batch, mask):
        feed_dict = {self.inputs: x_batch,
                     self.sequence_lengths: seq_lengths_batch,
                     self.mask: mask}
        preds_clip = self.sess.run(self.preds_clip, feed_dict=feed_dict)
        return preds_clip

    def _compute_accuracy(self, probs, targets, mask):
        pred_class = tf.cast(tf.argmax(probs, 1), tf.int32)
        hits = tf.reshape(
            tf.equal(pred_class, targets), [-1, self.max_sequence_length])
        total_hits = tf.reduce_sum(
            tf.cast(tf.logical_and(hits, mask), tf.float32),
            reduction_indices=1)
        return tf.reduce_mean(total_hits / tf.cast(self.sequence_lengths,
                                                   tf.float32))

    def _compute_accuracy_clip(self, probs, targets, mask):
        probs_r = tf.reshape(
            probs,
            [self.batch_size, self.max_sequence_length, self.num_classes])
        mask_expanded = tf.cast(tf.expand_dims(mask, 2), tf.float32)
        mask_repeated = tf.tile(mask_expanded, [1, 1, self.num_classes])
        seq_lengths_expanded = tf.cast(
            tf.expand_dims(self.sequence_lengths, 1), tf.float32)

        avg_probs = tf.reduce_sum(
            probs_r * mask_repeated,
            reduction_indices=1) / seq_lengths_expanded
        preds_clip = tf.cast(tf.argmax(avg_probs, 1), tf.int32)

        targets_sliced = tf.squeeze(
            tf.slice(targets, [0, 0], [-1, 1]), squeeze_dims=[1])
        return tf.reduce_mean(
            tf.cast(tf.equal(preds_clip, targets_sliced), tf.float32))

    def get_accuracy(self, x_batch, y_batch, seq_lengths_batch, mask):
        feed_dict = {self.inputs: x_batch,
                     self.targets: y_batch,
                     self.sequence_lengths: seq_lengths_batch,
                     self.mask: mask}
        accuracy_value = self.sess.run(self.accuracy, feed_dict=feed_dict)
        return accuracy_value

    def get_accuracy_clip(self, x_batch, y_batch, seq_lengths_batch, mask):
        feed_dict = {self.inputs: x_batch,
                     self.targets: y_batch,
                     self.sequence_lengths: seq_lengths_batch,
                     self.mask: mask}
        accuracy_value_clip = self.sess.run(self.accuracy_clip,
                                            feed_dict=feed_dict)
        return accuracy_value_clip

    def cost(self, x_batch, y_batch, seq_lengths_batch, mask):
        feed_dict = {self.inputs: x_batch,
                     self.targets: y_batch,
                     self.sequence_lengths: seq_lengths_batch,
                     self.mask: mask}
        cost_ = self.sess.run(self.total_cost, feed_dict=feed_dict)
        return cost_

    def train(self, x_batch, y_batch, seq_lengths_batch, mask):
        feed_dict = {self.inputs: x_batch,
                     self.targets: y_batch,
                     self.sequence_lengths: seq_lengths_batch,
                     self.mask: mask}
        cost_train, accuracy_train, accuracy_clip_train, summary_train, _ = self.sess.run(
            [self.total_cost, self.accuracy, self.accuracy_clip, self.merged,
             self.train_op],
            feed_dict=feed_dict)
        return cost_train, accuracy_train, accuracy_clip_train, summary_train

    def predict(self, x_batch, seq_lengths_batch, mask):
        feed_dict = {self.inputs: x_batch,
                     self.sequence_lengths: seq_lengths_batch,
                     self.mask: mask}
        softmax_r = tf.reshape(
            self.softmax,
            [self.batch_size, self.max_sequence_length, self.num_classes])
        predictions = self.sess.run(softmax_r, feed_dict=feed_dict)
        return predictions

    def val_batch(self, x_batch, y_batch, seq_lengths_batch, mask):
        feed_dict = {self.inputs: x_batch,
                     self.targets: y_batch,
                     self.sequence_lengths: seq_lengths_batch,
                     self.mask: mask}
        cost_val, accuracy_val, accuracy_clip_val, summary_val = self.sess.run(
            [self.total_cost, self.accuracy, self.accuracy_clip, self.merged],
            feed_dict=feed_dict)
        return cost_val, accuracy_val, accuracy_clip_val, summary_val

    def load(self, checkpoint):
        self.saver.restore(self.sess, checkpoint)

    def save(self, save_filename):
        # Save the variables to disk.
        save_path = self.saver.save(
            self.sess, os.path.join(self.save_path, save_filename) + '.ckpt')
        print("Model saved in file: %s" % save_path)
