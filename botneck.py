from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import math
from datetime import datetime
import pickle
import os
import scipy.misc

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.contrib.slim as slim

import util
import ops
import dataset_manager

tf_flags = tf.app.flags
FLAGS = tf_flags.FLAGS
tf_flags.DEFINE_boolean('dotrain', True, 'do train or test?')
tf_flags.DEFINE_string('train_dir', '', 'training_directory')
#tf_flags.DEFINE_float('disc_lr_coeff', 1.0, 'coefficient for discriminator learning rate')

class VAE(object):
    def __init__(self, batch_size, code_dim, img_encoder_params, img_decoder_params, images, div_loss):
        self.batch_size             = batch_size
        self.code_dim               = code_dim

        # Constants
        self.codes_prior_sigma      = 1.0
  
        # Inputs
        self.images = images
        self.div_loss = div_loss 

        # Placeholders
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
        self.stocha = tf.placeholder(tf.float32, [1])
        self.beta = tf.placeholder(tf.float32, [1])
        self.codes_noise = tf.placeholder(tf.float32, [self.batch_size, 1, 1, self.code_dim])
        self.prior_samples = tf.placeholder(tf.float32, [self.batch_size, 1, 1, self.code_dim])

        # Variables
        with tf.variable_scope("model") as scope:
            self.img_encoder = ops.ConvEncoder(**img_encoder_params)
            self.img_parameterizer = ops.GaussianParameterizer(self.code_dim, 'img_codes', ksize=1)

            self.img_decoder = ops.ConvDecoder(**img_decoder_params)

            self.prior_codes_mu = tf.zeros([self.batch_size, 1, 1, self.code_dim])
            self.prior_codes_sigma = self.codes_prior_sigma * tf.ones([self.batch_size, 1, 1, self.code_dim])

        # Summaries
        tf.summary.scalar('learning_rate', self.lr)
        tf.summary.scalar('stocha0', self.stocha[0])
        tf.summary.scalar('beta0', self.beta[0])

    def loss_graph(self):
        losses = []
        losses.append(self.div_loss(self.prior_samples, self.codes))
        losses.append(self.recon_loss())
        for l in losses:
            print(l)
            tf.add_to_collection('losses', l)
            tf.summary.scalar(l.name, l)

        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar(self.loss.name, self.loss)


    def recon_loss(self):
        diff = self.recs_mu - self.images
        diff = tf.square(diff)
        loss = tf.reduce_mean(tf.reduce_sum(diff, axis=(1,2,3)))
        loss = tf.identity(self.beta[0]*loss, name='recon_loss')
        return loss


    def train_graph(self, reuse=False):
        self.codes_mu, self.codes_sigma = self.encode(self.images, reuse=reuse)
        self.codes = ops.sample_gaussian(self.codes_mu, self.codes_sigma, self.codes_noise, 'sample_codes', self.stocha[0])

        self.recs_mu = self.decode_codes(self.codes, reuse=reuse)

        util.activation_summary(self.codes_mu, 'img_codes_mu')
        util.activation_summary(self.codes_sigma, 'img_codes_sigma')
        util.activation_summary(self.codes, 'img_codes')


    def sample_graph(self, reuse=False):
        self.sampled_codes = ops.sample_gaussian(self.prior_codes_mu, self.prior_codes_sigma, self.codes_noise, 'sample_codes', 1.0)

        self.sampled_recs_mu = self.decode_codes(self.sampled_codes, reuse=reuse)
        util.activation_summary(self.sampled_codes, 'sampled_img_codes')


    def encode(self, images, reuse=False):
        img_feats = self.img_encoder.encode(images, is_training=self.is_training, reuse=reuse)
        codes_mu, codes_sigma = self.img_parameterizer.get_params(img_feats, is_training=self.is_training, reuse=reuse)
        return codes_mu, codes_sigma


    def decode_codes(self, codes, reuse=False):
        _ = self.img_decoder.decode(codes, is_training=self.is_training, reuse=reuse)
        recs_mu = self.img_decoder.rec_stack[-1]
        return recs_mu

  
    def train(self, global_step):
        with tf.control_dependencies([self.loss]):
            return ops.train(self.loss, global_step, learning_rate=self.lr, name='training_step')


class RandDiv(object):

    def __init__(self, batch_size, code_dim, num_projsigs=256):
        # params
        self.batch_size = batch_size
        self.code_dim = code_dim
        self.num_projsigs = num_projsigs

        ## placeholders
        self.projsigs_vals = {}

        self.projsigs = {}
        self.projsigs['kernel'] = tf.placeholder(tf.float32, [1, 1, self.code_dim, self.num_projsigs])
        self.projsigs['bias'] = tf.placeholder(tf.float32, [1, 1, 1, self.num_projsigs])

        self.ema = tf.train.ExponentialMovingAverage(decay=0.95)

        with tf.variable_scope('RandDiv') as scope:
            self.projsigs['prior_pos'] = tf.get_variable('prior_pos', dtype=tf.float32, shape=[1, 1, 1, self.num_projsigs], trainable=False, initializer=tf.constant_initializer(0.5))
            self.projsigs['codes_pos'] = tf.get_variable('codes_pos', dtype=tf.float32, shape=[1, 1, 1, self.num_projsigs], trainable=False, initializer=tf.constant_initializer(0.5))

        tf.summary.scalar('prior_pos', self.projsigs['prior_pos'][0,0,0,0])
        tf.summary.scalar('codes_pos', self.projsigs['codes_pos'][0,0,0,0])
        # end __init__


    def next_epoch(self, sess):
        # get new values for projsigs
        kernel = (np.random.randn(1, 1, self.code_dim, self.num_projsigs))
        kernel = kernel/np.sum(np.square(kernel), axis=3, keepdims=True)
        self.projsigs_vals['kernel'] = kernel.astype(np.float32)
        bias = (np.random.randn(1, 1, 1, self.num_projsigs))
        self.projsigs_vals['bias'] = bias.astype(np.float32)
        tf.assign(self.projsigs['prior_pos'], 0.5*np.ones([1, 1, 1, self.num_projsigs], dtype=np.float32)).eval(session=sess)
        tf.assign(self.projsigs['codes_pos'], 0.5*np.ones([1, 1, 1, self.num_projsigs], dtype=np.float32)).eval(session=sess)


    def cur_feed_dict(self):
        feed_dict = {}
        feed_dict[self.projsigs['kernel']] = self.projsigs_vals['kernel']
        feed_dict[self.projsigs['bias']] = self.projsigs_vals['bias']
        return feed_dict


    def div_loss(self, prior_samples, codes):

        def get_logits(inp, cur_projsig):
            convout = tf.nn.conv2d(inp, cur_projsig['kernel'], strides=(1,1,1,1), padding='SAME')
            convshape = convout.get_shape().as_list()
            assert len(convshape)==4, 'inp must be 4 dimensional tensor'
            cur_logits = convout + cur_projsig['bias']
            return 1000.*cur_logits


        def update_stats():
            # feed through cur_learned_projsigs
            # prior samples stats
            samples_logits = get_logits(prior_samples, self.projsigs)
            samples_stats = tf.sigmoid(samples_logits)

            samples_bool_pos = tf.greater(samples_stats, tf.constant(0.5, dtype=tf.float32))
            samples_bool_pos = tf.cast(samples_bool_pos, tf.float32)
            samples_bool_pos = tf.reduce_mean(samples_bool_pos, axis=(0), keep_dims=True)

            # codes stats
            codes_logits = get_logits(codes, self.projsigs)
            codes_stats = tf.sigmoid(codes_logits)
            codes_stats = tf.reduce_mean(codes_stats, axis=(0), keep_dims=True)

            ema_apply_op = self.ema.apply([samples_bool_pos, codes_stats])
            prior_pos_op = tf.assign(self.projsigs['prior_pos'], self.ema.average(samples_bool_pos))
            codes_pos_op = tf.assign(self.projsigs['codes_pos'], self.ema.average(codes_stats))
            with tf.control_dependencies([ema_apply_op, prior_pos_op]):
                return tf.identity(self.projsigs['prior_pos']), tf.identity(0.5*codes_stats + 0.5*self.projsigs['codes_pos'])
            
        prior_pos, codes_pos = update_stats()

        loss = tf.square(prior_pos - codes_pos)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1,2,3)), name='div_loss')

        return loss


def train(train_dir):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        batch_size = 16
        code_dim = 128

        img_encoder_params = {
                        'scopename' : 'img_enc', 
                        'channels' : [32,32,64,128,256,512], 
                        'strides' :  [1, 2, 2, 2,  2,  4], # 64, 32, 16, 8, 4, 1
                        'ksizes' :   [3, 3, 3, 3,  3,  4],
                        'batch_norm' : True
                    }

        img_decoder_params = {
                        'scopename' : 'img_dec', 
                        'channels' :  [512, 256, 128, 64, 32, 32],
                        'ksizes' :    [4,   3,   3,   3,  3,  1],
                        'outshapes' : [4,   8,   16,  32, 64, 64],
                        'colorout' :  [0,   0,   0,   0,  0,  1],
                        'COLOR_CHN' : 3,
                        'outlin' : False,
                        'batch_norm' : True
                    }

        dataset = dataset_manager.get_dataset('celeba64')
        image_size, color_chn = dataset.train.get_dims()
        train_images = dataset.train.next_batch(batch_size)

        num_steps = math.ceil(dataset.train.num_img/batch_size)
        num_epochs = 100

        randdiv = RandDiv(batch_size, code_dim)

        model = VAE(batch_size, code_dim, img_encoder_params, img_decoder_params, train_images, randdiv.div_loss)
        model.train_graph()
        model.sample_graph(reuse=True)

        model.loss_graph()
        model_train_op = model.train(global_step)

        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        util.image_summary(model.images, 'model_inp')
        util.image_summary(model.recs_mu, 'model_rec')
        util.image_summary(model.sampled_recs_mu, 'sample')
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        stocha0 = 1.0
        beta0 = 1.0

        summary_step = 0
        cur_lr = 0.0001
        for epoch in xrange(num_epochs):
            if epoch%30 == 15:
                cur_lr = cur_lr/10.
                

            randdiv.next_epoch(sess)

            cur_feed_dict = randdiv.cur_feed_dict()
            cur_feed_dict[model.lr] = cur_lr
            cur_feed_dict[model.is_training] = True
            cur_feed_dict[model.stocha] = np.array([stocha0])
            cur_feed_dict[model.beta] = np.array([beta0])

            for step in xrange(num_steps):
                cur_feed_dict[model.codes_noise] = np.random.randn(batch_size, 1, 1, code_dim).astype('float32')
                cur_feed_dict[model.prior_samples] = np.random.randn(batch_size, 1, 1, code_dim).astype('float32')

                _, model_loss_val = sess.run([model_train_op, model.loss], feed_dict=cur_feed_dict)

                if step%200==0 or (step + 1) == num_steps:
                    format_str = ('%s: epoch %d of %d, step %d of %d, model_loss = %.5f')
                    print (format_str % (datetime.now(), epoch, num_epochs-1, step, num_steps-1, model_loss_val))

                if step%200==0 or (step + 1) == num_steps:
                    summary_str = sess.run(summary_op, feed_dict=cur_feed_dict)
                    summary_writer.add_summary(summary_str, summary_step)
                    summary_step += 1

                # Save the model checkpoint periodically.
                if (epoch%5==0 or (epoch+1)==num_epochs) and (step + 1) == num_steps:
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=(epoch))

                if (step + 1) == num_steps:
                    recs_mu, cur_images = sess.run([model.recs_mu, model.images], feed_dict=cur_feed_dict)
                    for i in xrange(min(20,batch_size)):
                        tmp = np.concatenate([cur_images[i,:,:,:], recs_mu[i,:,:,:]], axis=0)
                        util.save_img(tmp, os.path.join(train_dir, 'epoch_%d_img_%d.png' % (epoch, i)))


        coord.request_stop()
        coord.join(threads)


def test(test_dir, train_dir):
    with tf.Graph().as_default():
        batch_size = 16
        code_dim = 128

        img_encoder_params = {
                        'scopename' : 'img_enc', 
                        'channels' : [32,32,64,128,256,512], 
                        'strides' :  [1, 2, 2, 2,  2,  4], # 64, 32, 16, 8, 4, 1
                        'ksizes' :   [3, 3, 3, 3,  3,  4],
                        'batch_norm' : True
                    }

        img_decoder_params = {
                        'scopename' : 'img_dec', 
                        'channels' :  [512, 256, 128, 64, 32, 32],
                        'ksizes' :    [4,   3,   3,   3,  3,  1],
                        'outshapes' : [4,   8,   16,  32, 64, 64],
                        'colorout' :  [0,   0,   0,   0,  0,  1 ],
                        'COLOR_CHN' : 3,
                        'outlin' : False,
                        'batch_norm' : True
                    }

        dataset = dataset_manager.get_dataset('celeba64')
        image_size, color_chn = dataset.train.get_dims()
        test_images = dataset.test.next_batch(batch_size, doperm=False)

        num_steps = 4#math.ceil(dataset.test.num_img/batch_size)

        model = VAE(batch_size, code_dim, img_encoder_params, img_decoder_params, test_images, None)
        model.train_graph()
        model.sample_graph(reuse=True)

        saver = tf.train.Saver(tf.global_variables())

        # Start running operations on the Graph.
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # TODO Initialize variables?

        # Load variables 
        ckpt = tf.train.get_checkpoint_state(train_dir)
        print('loading from', train_dir)

        if ckpt and ckpt.model_checkpoint_path:
            vars_to_restore = tf.global_variables()
            res_saver = tf.train.Saver(vars_to_restore)
            
            # Restores from checkpoint
            model_checkpoint_path = os.path.abspath(ckpt.model_checkpoint_path)
            print(model_checkpoint_path)
            res_saver.restore(sess, model_checkpoint_path)
        else:
            print('Error: no checkpoint file found')
            exit()

        stocha0 = 1.0
        beta0 = 1.0
            
        cur_feed_dict={}
        cur_feed_dict[model.is_training] = False
        cur_feed_dict[model.stocha] = np.array([stocha0])
        cur_feed_dict[model.beta] = np.array([beta0])

        for step in xrange(num_steps):
            cur_feed_dict[model.codes_noise] = np.random.randn(batch_size, 1, 1, code_dim).astype('float32')
            cur_feed_dict[model.prior_samples] = np.random.randn(batch_size, 1, 1, code_dim).astype('float32')

            recs_mu, cur_images = sess.run([model.recs_mu, model.images], feed_dict=cur_feed_dict)
            for i in xrange(batch_size):
                tmp = np.concatenate([cur_images[i,:,:,:], recs_mu[i,:,:,:]], axis=0)
                util.save_img(tmp, os.path.join(test_dir, 'img_%d.png' % (step*batch_size + i)))

            if step%100==0 or (step + 1) == num_steps:
                format_str = ('%s: step %d of %d')
                print (format_str % (datetime.now(), step, num_steps-1))



        coord.request_stop()
        coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.dotrain==True:
        train_dir = FLAGS.train_dir
        if len(train_dir)==0:
            train_dir = 'logs_loss/'

        print('Training model at', train_dir)

        if tf.gfile.Exists(train_dir):
            tf.gfile.DeleteRecursively(train_dir)
        os.makedirs(train_dir)
        train(train_dir)

    elif FLAGS.dotrain==False:
        train_dir = FLAGS.train_dir
        test_dir = 'test_' + train_dir

        if tf.gfile.Exists(test_dir):
            tf.gfile.DeleteRecursively(test_dir)
        os.makedirs(test_dir)

        print('Testing model at', train_dir)
        test(test_dir, train_dir)


if __name__ == '__main__':
    tf.app.run()

