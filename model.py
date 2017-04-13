from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import math
from datetime import datetime
import pickle
import os
import scipy.misc

from six.moves import xrange  # pylint: disable=redefined-builtin

import util
import ops
import dataset_manager

epsilon = 0.0000001

class VAE(object):
    def __init__(self, batch_size, code_dim, img_encoder_params, img_decoder_params, images, eval_loss):

        self.batch_size             = batch_size
        self.code_dim               = code_dim

        # Constants
        self.codes_prior_sigma      = 1.0
        self.hard_gumbel            = False
  
        # Inputs
        self.images = images
        self.eval_loss = eval_loss

        # Placeholders
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
        self.stocha = tf.placeholder(tf.float32, [1])
        self.beta = tf.placeholder(tf.float32, [1])
        self.codes_noise = tf.placeholder(tf.float32, [self.batch_size, 1, 1, self.code_dim])

        # Variables
        with tf.variable_scope("model") as scope:
            self.img_encoder = ops.ConvEncoder(**img_encoder_params)
            self.img_parameterizer = ops.GaussianParameterizer(self.img_encoder.outdim(), self.code_dim, 'img_codes', ksize=1)

            self.img_decoder = ops.ConvDecoder(**img_decoder_params)

            self.prior_codes_mu = tf.zeros([self.batch_size, 1, 1, self.code_dim])
            self.prior_codes_sigma = self.codes_prior_sigma * tf.ones([self.batch_size, 1, 1, self.code_dim])

        # Summaries
        tf.summary.scalar('learning_rate', self.lr)
        tf.summary.scalar('stocha0', self.stocha[0])
        tf.summary.scalar('beta0', self.beta[0])

    def loss_graph(self):
        losses = self.eval_loss(self.recs_mu)
        print(losses)
        losses.append(self.kldiv_loss())
        for l in losses:
            print(l)
            tf.add_to_collection('losses', l)
            tf.summary.scalar(l.name, l)

        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar(self.loss.name, self.loss)


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

  
    def kldiv_loss(self):
        # kl-divergence 
        codes_mu = tf.reshape(self.codes_mu, [-1, 1, 1, self.code_dim])
        codes_sigma = tf.reshape(self.codes_sigma, [-1, 1, 1, self.code_dim])
        code_kldiv = ops.kldiv_unitgauss(codes_mu, codes_sigma, coeff=self.beta[0])
        code_kldiv_loss = tf.reduce_mean(code_kldiv, name='code_kldiv')
        return code_kldiv_loss

    
    
    def train(self, global_step):
        with tf.control_dependencies([self.loss]):
            return ops.train(self.loss, global_step, learning_rate=self.lr, name='training_step')


class BalancedLoss(object):

    def __init__(self, batch_size, image_size, color_chn, images, ksize=1, num_projsigs=10):
        # params
        self.batch_size = batch_size
        self.image_size = image_size
        self.color_chn = color_chn
        self.ksize = ksize
        self.num_projsigs = num_projsigs

        # input
        self.images = images

        # placeholders
        self.kernels = []
        self.biases = []
        self.y_targets = []
        self.pos_weights = []
        for i in xrange(self.num_projsigs):
            self.kernels.append(tf.placeholder(tf.float32, [self.ksize, self.ksize, self.color_chn, 1]))
            self.biases.append(tf.placeholder(tf.float32, [1]))
            self.y_targets.append(tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 1]))
            self.pos_weights.append(tf.placeholder(tf.float32, [1]))

        self.learn_loss = self.get_learn_loss()


    def cur_feed_dict(self):
        # TODO
        retval = {}
        for k in self.kernels:
            retval[k] = np.random.randn([self.ksize, self.ksize, self.color_chn, 1], dtype=np.float32)
        for b in self.biases:
            retval[b] = np.random.randn([1], dtype=np.float32) + 0.5
        for yt in self.y_targets:
            retval[yt] = np.random.randint(2, size=[self.batch_size, self.image_size, self.image_size, 1], dtype=np.float32)
        for pw in self.pos_weights:
            retval[pw] = np.random.rand(1).astype(np.float32)
        return retval


    def get_learn_loss(self):
        pass


    def eval_loss(self, inp):
        losses = []
        for i in xrange(self.num_projsigs):
            convout = tf.nn.conv2d(inp, self.kernels[i], strides=(1,1,1,1), padding='SAME')
            cur_logits = tf.nn.bias_add(convout, self.biases[i])

            cur_loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.y_targets[i], logits=cur_logits, pos_weight=self.pos_weights[i])

            cur_loss = tf.reduce_sum(cur_loss, axis=(1,2,3))
            cur_loss = tf.reduce_mean(cur_loss, name='wcel'+str(i))
            losses.append(cur_loss)
        return losses


def train(train_dir):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        batch_size = 32
        code_dim = 64

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
                        'ksizes' :    [4,   3,   3,   3,  3,  1 ],
                        'outshapes' : [4,   8,   16,  32, 64, 64],
                        'colorout' :  [0,   0,   0,   0,  0,  1 ],
                        'COLOR_CHN' : 3,
                        'outlin' : False,
                        'batch_norm' : True
                    }

        dataset = dataset_manager.get_dataset('celeba64')
        image_size, color_chn = dataset.train.get_dims()
        train_images = dataset.train.next_batch(batch_size)

        balanced_loss = BalancedLoss(batch_size, image_size, color_chn, train_images)

        model = VAE(batch_size, code_dim, img_encoder_params, img_decoder_params, train_images, balanced_loss.eval_loss)
        model.train_graph()
        model.sample_graph(reuse=True)

        model.loss_graph()
        train_op = model.train(global_step)

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

        num_steps = math.ceil(dataset.train.num_img/batch_size)
        num_epochs = 30

        stocha0 = 0.0
        beta0 = 1.00

        summary_step = 0
        cur_lr = 0.001
        for epoch in xrange(num_epochs):
            if epoch%15 == 14:
                cur_learning_rate = cur_learning_rate/10.
                
            epoch_feed_dict = balanced_loss.cur_feed_dict()
            for step in xrange(num_steps):
                # TODO 
                cur_feed_dict = epoch_feed_dict.copy()
                cur_feed_dict[model.lr] = cur_lr
                cur_feed_dict[model.is_training] = True
                cur_feed_dict[model.stocha] = np.array([stocha0])
                cur_feed_dict[model.beta] = np.array([beta0])
                cur_feed_dict[model.codes_noise] = np.random.randn(batch_size, 1, 1, code_dim).astype('float32')

                _ = sess.run(train_op, feed_dict=cur_feed_dict)

                model_loss_val = sess.run(model.model_loss, feed_dict=cur_feed_dict)

                if step%2==0 or (step + 1) == num_steps:
                    format_str = ('%s: epoch %d of %d, step %d of %d, model_loss = %.5f')
                    print (format_str % (datetime.now(), epoch, num_epochs-1, step, num_steps-1, model_loss_val))

                if step%10==0:
                    summary_str = sess.run(summary_op, feed_dict=cur_feed_dict)
                    summary_writer.add_summary(summary_str, summary_step)
                    summary_step += 1

                # Save the model checkpoint periodically.
                if (epoch%2==0 or (epoch+1)==num_epochs) and (step + 1) == num_steps:
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=(step))

        coord.request_stop()
        coord.join(threads)



def main(argv=None):  # pylint: disable=unused-argument
    train_dir = 'logs/'

    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)

    os.makedirs(train_dir)
    train(train_dir)


if __name__ == '__main__':
    tf.app.run()

