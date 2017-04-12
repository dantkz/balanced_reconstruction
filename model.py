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

epsilon = 0.0000001

class VAE(object):
    def __init__(self, batch_size, code_dim, img_encoder_params, img_decoder_params, images):

        self.batch_size             = batch_size
        self.code_dim               = code_dim
        self.IMAGE_SIZE             = img_decoder_params['outshapes'][-1]
        self.COLOR_CHN              = img_decoder_params['COLOR_CHN']

        # Constants
        self.codes_prior_sigma      = 1.0
        self.hard_gumbel            = False
  
        # Inputs
        self.images = images

        # Placeholders
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
        self.stocha = tf.placeholder("float", [1])
        self.beta = tf.placeholder("float", [1])
        self.codes_noise = tf.placeholder("float", [self.batch_size, 1, 1, self.code_dim])

        # Variables
        with tf.variable_scope("model") as scope:
            self.img_encoder = ops.ConvEncoder(img_encoder_params)
            self.img_parameterizer = ops.GaussianParameterizer(self.img_encoder.outdim(), self.code_dim, 'img_codes', ksize=1)

            self.img_decoder = ops.ConvDecoder(img_decoder_params)

            self.raw_sigma = ops.variable('raw_sigma', shape=[1,1], initializer=tf.constant_initializer(1.0))
            self.prior_codes_mu = tf.zeros([self.batch_size, 1, 1, self.code_dim])
            self.prior_codes_sigma = self.codes_prior_sigma * tf.ones([self.batch_size, 1, 1, self.code_dim])

        self.sigma = tf.reshape(tf.nn.softplus(tf.reshape(self.raw_sigma, [1, 1, 1, 1])), [1])

        # Summaries
        util.activation_summary(self.sigma, 'sigma')
        tf.summary.scalar('learning_rate', self.lr)
        tf.summary.scalar('sigma', self.sigma[0])
        tf.summary.scalar('stocha0', self.stocha[0])
        tf.summary.scalar('beta0', self.beta[0])

    def loss_graph(self):
        losses = self.get_losses()
        for l in losses:
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
        recs_mu = self.img_decoder.decode(codes, is_training=self.is_training, reuse=reuse)
        return recs_mu

  
    def get_losses(self):
        losses = []
        def get_recon_loss(images, recs_mu, scale_factor, name):
            recon_diff = tf.square(images - recs_mu)
            recon_loss = scale_factor * 0.5 * tf.reduce_sum( \
                    2.0*tf.log(self.sigma) \
                       + tf.div(recon_diff, tf.square(tf.maximum(epsilon, self.sigma))) \
                       , [1,2,3])
            recon_loss = tf.multiply(1.0/(self.batch_size), tf.reduce_sum(recon_loss), name='recon_loss' + name)
            return recon_loss



        # kl-divergence 
        codes_mu = tf.reshape(self.codes_mu, [-1, 1, 1, self.code_dim])
        codes_sigma = tf.reshape(self.codes_sigma, [-1, 1, 1, self.code_dim])
        code_kldiv = ops.kldiv_unitgauss(codes_mu, codes_sigma, coeff=self.beta[0])
        code_kldiv_loss = tf.reduce_mean(code_kldiv, name='code_kldiv')
        losses.append(code_kldiv_loss)

        return losses

    
    
    def train(self, global_step):
        with tf.control_dependencies([self.loss]):
            return ops.train(self.loss, global_step, learning_rate=self.lr, name='training_step')



def train(train_dir):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

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
        batch_size = 32
        train_images = dataset.train.next_batch(batch_size)
        model = VAE(batch_size, 64, img_encoder_params, img_decoder_params, train_images)
        model.train_graph()
        model.sample_graph(reuse=True)

        model.loss_graph()
        train_op = model.train(global_step)

        # TODO bloss_train

        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        for i, cur_image in enumerate(model.images_stack):
            util.image_summary(cur_image, str(i) + '_' + 'inp')
        for i, cur_recs in enumerate(model.recs_mu_stack):
            util.image_summary(cur_recs, str(i) + '_rec')
        for i, cur_recs in enumerate(model.sampled_recs_mu_stack):
            util.image_summary(cur_recs, 'sampled_' + str(i))
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        num_steps = math.ceil(flags.num_examples/flags.batch_size)
        num_epochs = 30

        stocha0 = 0.0
        beta0 = 1.00

        summary_step = 0
        cur_lr = 0.001
        for epoch in xrange(num_epochs):
            if epoch%15 == 14:
                cur_learning_rate = cur_learning_rate/10.
                
            for step in xrange(num_steps):
                cur_feed_dict = {}
                cur_feed_dict[model.lr] = cur_lr
                cur_feed_dict[model.is_training] = True
                cur_feed_dict[model.stocha] = np.array([stocha0])
                cur_feed_dict[model.beta] = np.array([beta0])
                cur_feed_dict[model.codes_noise] = np.random.randn(flags.batch_size, 1, 1, flags.code_dim).astype('float32')

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

