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
tf_flags.DEFINE_integer('loss_type', -1, '0: balanced cross-entropy with sigmoids loss, 1: l1 of projections of reconstructions loss, 2: l1 reconstructions loss, 3: l2 of reconstructions loss')
tf_flags.DEFINE_string('train_dir', '', 'training_directory')
#tf_flags.DEFINE_boolean('round_real', True, 'round real data')
#tf_flags.DEFINE_integer('dim', -1, '')
#tf_flags.DEFINE_float('disc_lr_coeff', 1.0, 'coefficient for discriminator learning rate')

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
            self.img_parameterizer = ops.GaussianParameterizer(self.code_dim, 'img_codes', ksize=1)

            self.img_decoder = ops.ConvDecoder(**img_decoder_params)

            self.prior_codes_mu = tf.zeros([self.batch_size, 1, 1, self.code_dim])
            self.prior_codes_sigma = self.codes_prior_sigma * tf.ones([self.batch_size, 1, 1, self.code_dim])

        # Summaries
        tf.summary.scalar('learning_rate', self.lr)
        tf.summary.scalar('stocha0', self.stocha[0])
        tf.summary.scalar('beta0', self.beta[0])

    def loss_graph(self):
        losses = self.eval_loss(self.recs_mu, self.images)
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

    def __init__(self, batch_size, image_size, color_chn, images, num_steps, image_scales, ksize=3, num_projsigs=64):
        # params
        self.batch_size = batch_size
        self.image_size = image_size
        self.color_chn = color_chn
        self.ksize = ksize
        self.num_projsigs = num_projsigs
        self.num_steps = num_steps
        self.image_scales = image_scales

        # input
        self.images = images

        # placeholders
        self.cur_eval_projsigs = []
        self.cur_learned_projsigs = []
        self.eval_placeholders = []

        for cur_image_size in self.image_scales:
            self.cur_eval_projsigs.append({})
            self.cur_learned_projsigs.append({}) 
            self.eval_placeholders.append({}) 

            self.eval_placeholders[-1]['kernel'] = tf.placeholder(tf.float32, [self.ksize, self.ksize, self.color_chn, self.num_projsigs])
            self.eval_placeholders[-1]['bias'] = tf.placeholder(tf.float32, [1, 1, 1, self.num_projsigs])
            self.eval_placeholders[-1]['pos_weight'] = tf.placeholder(tf.float32, [1, 1, 1, self.num_projsigs])


            # Variables
            self.cur_learned_projsigs[-1]['kernel'] = 0.00001*np.ones([self.ksize, self.ksize, self.color_chn, self.num_projsigs], dtype=np.float32)
            self.cur_learned_projsigs[-1]['bias'] = (0.00001*np.ones([1, 1, 1, self.num_projsigs])).astype(np.float32)

            for ch in xrange(self.color_chn):
                start = int(ch*self.num_projsigs/self.color_chn)
                end = int((ch+1)*self.num_projsigs/self.color_chn)
                self.cur_learned_projsigs[-1]['kernel'][self.ksize//2, self.ksize//2, ch, start:end] = 1.
                self.cur_learned_projsigs[-1]['bias'][0,0,0,start:end] = np.linspace(-1.1, 0.1, end-start, dtype=np.float32)

            with tf.variable_scope('BalancedLoss') as scope:
                self.cur_learned_projsigs[-1]['pos'] = tf.get_variable('pos_'+str(cur_image_size), dtype=tf.float32, shape=[1, 1, 1, self.num_projsigs], trainable=False, initializer=tf.constant_initializer(1.0))
                self.cur_learned_projsigs[-1]['neg'] = tf.get_variable('neg_'+str(cur_image_size), dtype=tf.float32, shape=[1, 1, 1, self.num_projsigs], trainable=False, initializer=tf.constant_initializer(1.0))

        # end __init__


    def next_epoch(self, sess):
        # push learned values from cur_learned_projsigs to cur_eval_projsigs
        for scale_idx, _ in enumerate(self.image_scales):
            #self.cur_eval_projsigs[scale_idx] = {} # TODO
            self.cur_eval_projsigs[scale_idx]['kernel'] = self.cur_learned_projsigs[scale_idx]['kernel']
            self.cur_eval_projsigs[scale_idx]['bias'] = self.cur_learned_projsigs[scale_idx]['bias']
            pos = self.cur_learned_projsigs[scale_idx]['pos'].eval(session=sess)
            neg = self.cur_learned_projsigs[scale_idx]['neg'].eval(session=sess)
            pnsum = pos + neg
            pos /= 0.00001+pnsum
            neg /= 0.00001+pnsum
            self.cur_eval_projsigs[scale_idx]['pos_weight'] = pos

        # get new values for cur_learned_projsigs
        for scale_idx, _ in enumerate(self.image_scales):
            self.cur_learned_projsigs[scale_idx]['kernel'] = (np.random.randn(self.ksize, self.ksize, self.color_chn, self.num_projsigs)).astype(np.float32)
            self.cur_learned_projsigs[scale_idx]['bias'] = (np.random.randn(1, 1, 1, self.num_projsigs)).astype(np.float32)
            tf.assign(self.cur_learned_projsigs[scale_idx]['pos'], np.zeros([1, 1, 1, self.num_projsigs], dtype=np.float32)).eval(session=sess)
            tf.assign(self.cur_learned_projsigs[scale_idx]['neg'], np.zeros([1, 1, 1, self.num_projsigs], dtype=np.float32)).eval(session=sess)


    def cur_feed_dict(self):
        feed_dict = {}
        for scale_idx, _ in enumerate(self.image_scales):
            feed_dict[self.eval_placeholders[scale_idx]['kernel']] = self.cur_eval_projsigs[scale_idx]['kernel']
            feed_dict[self.eval_placeholders[scale_idx]['bias']] = self.cur_eval_projsigs[scale_idx]['bias']
            feed_dict[self.eval_placeholders[scale_idx]['pos_weight']] = self.cur_eval_projsigs[scale_idx]['pos_weight']
        return feed_dict


    def eval_loss(self, recon, target):
        losses = []

        def get_logits(inp, cur_projsig):
            convout = tf.nn.conv2d(inp, cur_projsig['kernel'], strides=(1,1,1,1), padding='SAME')
            convshape = convout.get_shape().as_list()
            assert len(convshape)==4, 'inp must be 4 dimensional tensor'
            cur_logits = convout + cur_projsig['bias']
            return cur_logits

        def weighted_cross_entropy_with_logits(targets, logits, pos_weight):
            z = targets
            x = logits
            q = pos_weight
            qz = q*z
            l = 1. - q - z + qz
            val = l * x + (qz + l) * (tf.log1p(tf.exp(-tf.abs(x))) + tf.nn.relu(-x))
            return val


        # feed through cur_learned_projsigs
        for scale_idx, cur_image_size in enumerate(self.image_scales):
            cur_target = slim.avg_pool2d(target, self.image_size//cur_image_size, stride=self.image_size//cur_image_size, padding='SAME')
            cur_recon= slim.avg_pool2d(recon, self.image_size//cur_image_size, stride=self.image_size//cur_image_size, padding='SAME')

            ylogits = get_logits(cur_target, self.cur_learned_projsigs[scale_idx])
            ytargets = tf.sigmoid(ylogits)
            #ytargets = tf.round(ytargets)
            ytargets_bool = tf.greater(ytargets, tf.constant(0.5, dtype=tf.float32))
            ytargets_bool = tf.cast(ytargets_bool, tf.float32)

            update_pos = tf.assign_add(self.cur_learned_projsigs[scale_idx]['pos'], tf.reduce_mean(ytargets_bool, axis=(0,1,2), keep_dims=True))

            update_neg = tf.assign_add(self.cur_learned_projsigs[scale_idx]['neg'], tf.reduce_mean(1.-ytargets_bool, axis=(0,1,2), keep_dims=True))

            with tf.control_dependencies([update_pos, update_neg]):
                # feed through eval_placeholders
                cur_logits = get_logits(cur_recon, self.eval_placeholders[scale_idx])
                cur_labels = tf.sigmoid(cur_logits)

                ylogits = get_logits(cur_target, self.eval_placeholders[scale_idx])
                ytargets = tf.sigmoid(ylogits)
                #ytargets = tf.round(ytargets)

                if FLAGS.loss_type==0:
                    cur_loss = weighted_cross_entropy_with_logits(
                            targets=ytargets, 
                            logits=cur_logits, 
                            pos_weight=self.eval_placeholders[scale_idx]['pos_weight']
                    )
                elif FLAGS.loss_type==1:
                    cur_loss = tf.abs(ylogits-cur_logits)
                elif FLAGS.loss_type==2:
                    cur_loss = tf.abs(cur_recon-cur_target)
                elif FLAGS.loss_type==3:
                    cur_loss = tf.square(cur_recon-cur_target)
                else:
                    print('undefined loss type')
                    exit()

                cur_loss = tf.reduce_mean(tf.reduce_sum(cur_loss, axis=(1,2,3)), name='wcel_'+str(scale_idx))
                losses.append(cur_loss)

        return losses


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
                        'colorout' :  [0,   0,   0,   0,  0,  1 ],
                        'COLOR_CHN' : 3,
                        'outlin' : False,
                        'batch_norm' : True
                    }

        dataset = dataset_manager.get_dataset('celeba64')
        image_size, color_chn = dataset.train.get_dims()
        train_images = dataset.train.next_batch(batch_size)

        num_steps = math.ceil(dataset.train.num_img/batch_size)
        num_epochs = 100

        balanced_loss = BalancedLoss(batch_size, image_size, color_chn, train_images, num_steps, [64, 32, 16], ksize=3)

        model = VAE(batch_size, code_dim, img_encoder_params, img_decoder_params, train_images, balanced_loss.eval_loss)
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
                
            balanced_loss.next_epoch(sess)
            cur_feed_dict = balanced_loss.cur_feed_dict()
            cur_feed_dict[model.lr] = cur_lr
            cur_feed_dict[model.is_training] = True
            cur_feed_dict[model.stocha] = np.array([stocha0])
            cur_feed_dict[model.beta] = np.array([beta0])

            for step in xrange(num_steps):
                cur_feed_dict[model.codes_noise] = np.random.randn(batch_size, 1, 1, code_dim).astype('float32')

                _, model_loss_val = sess.run([model_train_op, model.loss], feed_dict=cur_feed_dict)

                if step%500==0 or (step + 1) == num_steps:
                    format_str = ('%s: epoch %d of %d, step %d of %d, model_loss = %.5f')
                    print (format_str % (datetime.now(), epoch, num_epochs-1, step, num_steps-1, model_loss_val))

                if step%500==0 or (step + 1) == num_steps:
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
            train_dir = 'logs_loss' + str(FLAGS.loss_type) + '/'

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

