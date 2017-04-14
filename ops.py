import tensorflow as tf

class ConvDecoder(object):
    
    def __init__(self, scopename, channels, ksizes, outshapes, outlin=False, batch_norm=False, COLOR_CHN=None, colorout=None):
        self.scopename = scopename
        self.colorout = colorout
        self.batch_norm = batch_norm
        self.COLOR_CHN = COLOR_CHN
        self.outlin = outlin

        self.ksizes = ksizes
        self.outshapes = outshapes
        self.channels = channels

        self._outdim = self.channels[-1]
        self.num_layers = len(self.channels)

        print(self.scopename)
        print(" num_layers:", self.num_layers)
        print(" channels:", self.channels)
        print(" batch_norm:", self.batch_norm)
        print(" ksizes:", self.ksizes)
        print(" outshapes:", self.outshapes)
        print(" COLOR_CHN:", self.COLOR_CHN)

    def decode(self, code, is_training, reuse=False):
        self.make_dec_stack(code, is_training=is_training, reuse=reuse)
        return self.dec_stack[-1]

    def outdim(self, i=None):
        if i==None:
            return self._outdim
        else:
            return self.channels[i]

    def make_dec_stack(self, code, is_training, reuse=False):
        dec_stack = []
        rec_stack = []
        inp = code
        prev_color = None

        for layer_i in range(self.num_layers):
            scopename = self.scopename + '_' + str(layer_i)

            inpshape = inp.get_shape().as_list()[1]
            inpdim = inp.get_shape().as_list()[-1]
            outshape = self.outshapes[layer_i]
            ksize = self.ksizes[layer_i]
            if outshape<ksize:
                ksize = outshape

            nonlin = tf.nn.elu
            dobn = False
            outdim = self.channels[layer_i]
            
            if self.outlin and layer_i+1==self.num_layers:
                nonlin = tf.identity
            if self.batch_norm and layer_i+1!=self.num_layers:
                dobn = True

            if inpshape==1:
                out = upconvlayer_tr(scopename, inp, ksize, inpdim, outdim, outshape, stride=outshape, reuse=reuse, nonlin=nonlin, dobn=dobn, padding='SAME', is_training=is_training)
            else:
                # conv
                if outshape>inpshape:
                    inp = tf.image.resize_nearest_neighbor(inp, [outshape, outshape])

                if prev_color is not None:
                    prev_rec = tf.image.resize_bilinear(prev_color, [outshape, outshape])
                    inp = tf.concat([inp, prev_rec], 3)
                    inpdim += self.COLOR_CHN

                out = convlayer(scopename, inp, ksize, inpdim, outdim, stride=1, reuse=reuse, nonlin=nonlin, dobn=dobn, is_training=is_training)

            dec_stack.append(out)

            if self.colorout and self.colorout[layer_i]>0:
                print(outshape)
                rec = get_color_mu(scopename, out, self.COLOR_CHN, reuse=reuse)
                rec_stack.append(rec)
                prev_color = rec
            else:
                prev_color = None

            inp = out

        self.dec_stack = dec_stack
        self.rec_stack = rec_stack


class ConvEncoder(object):
    
    def __init__(self, scopename, channels, strides, ksizes, batch_norm=False, outlin=False):
        self.scopename = scopename
        self.outlin = outlin
        self.batch_norm = batch_norm

        self.channels = channels
        self.strides = strides
        self.ksizes = ksizes
        self.num_layers = len(self.channels)

        print(self.scopename)
        print(" batch_norm:", self.batch_norm)
        print(" num_layers:", self.num_layers)
        print(" channels:", self.channels)
        print(" strides:", self.strides)
        print(" ksizes:", self.ksizes)

    def outdim(self):
        return self.channels[-1]

    def encode(self, image, is_training, reuse=False):
        inp = image
        for layer_i in range(self.num_layers):
            # conv
            scopename = self.scopename + '_' + str(layer_i)
            ksize = self.ksizes[layer_i]
            inpshape = inp.get_shape().as_list()[1]
            inpdim = inp.get_shape().as_list()[-1]
            if inpshape<ksize:
                ksize = inpshape

            nonlin = tf.nn.elu
            dobn = False
            padding = 'SAME'
            outdim = self.channels[layer_i]
            stride = self.strides[layer_i]
            
            if self.outlin and layer_i+1==self.num_layers:
                nonlin = tf.identity
            if self.batch_norm and layer_i+1!=self.num_layers:
                dobn = True
            if inpshape==ksize and layer_i+1==self.num_layers:
                padding = 'VALID'

            out = convlayer(scopename, inp, ksize, inpdim, outdim, stride=stride, reuse=reuse, nonlin=nonlin, dobn=dobn, padding=padding, is_training=is_training)

            inp = out
    
        return out



def variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False), trainable=True):
    #tf.constant_initializer(0.0)
    #tf.random_normal_initializer(stddev=0.0001)
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def convlayer(name, inp, ksize, inpdim, outdim, stride, reuse, nonlin=tf.nn.elu, dobn=True, padding='SAME', is_training=True, add_bias=True):
    scopename = 'conv_' + name
    print('', scopename)
    print('  inp:', inp)
    strides = [1, stride, stride, 1]
    with tf.variable_scope(scopename) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = variable(scopename + '_kernel', [ksize, ksize, inpdim, outdim])
        linout = tf.nn.conv2d(inp, kernel, strides=strides, padding=padding)
        if add_bias:
            inpsize = inp.get_shape().as_list()
            inpsize[-1] = 1
            inp0 = tf.zeros(inpsize, dtype=tf.float32)
            bias = variable(scopename + '_bias', [ksize, ksize, 1, outdim], tf.constant_initializer(0.00001))
            linout_bias = tf.nn.conv2d(inp0, bias, strides=strides, padding=padding)
            linout = linout + linout_bias
            #bias = variable(scopename + '_bias', [outdim], tf.constant_initializer(0.00001))
            #linout = tf.nn.bias_add(linout, bias)
        if dobn:
            bnout = batchnorm(linout, is_training, reuse=reuse)
        else:
            bnout = linout
        out = nonlin(bnout, name=scopename + '_nonlin')
    print('  out:', out)
    return out

def upconvlayer_tr(i, inp, ksize, inpdim, outdim, outshape, stride, reuse, nonlin=tf.nn.elu, dobn=True, padding='SAME', is_training=True, add_bias=True):
    scopename = 'uconvtr_' + str(i)
    print('', scopename)
    print('  inp:', inp)
    output_shape = [inp.get_shape().as_list()[0], outshape, outshape, outdim]
    strides = [1, stride, stride, 1]
    with tf.variable_scope(scopename) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = variable(scopename + '_kernel', [ksize, ksize, outdim, inpdim])
        linout = tf.nn.conv2d_transpose(inp, kernel, output_shape, strides=strides, padding=padding)
        if add_bias:
            bias = variable(scopename + '_bias', [outdim], tf.constant_initializer(0.0))
            linout = tf.nn.bias_add(linout, bias)
        if dobn:
            bnout = batchnorm(linout, is_training, reuse=reuse)
        else:
            bnout = linout
        out = nonlin(bnout, name=scopename + 'nonlin')
    print('  out:', out)
    return out



def get_color_mu(name, inp, outdim, ksize=1, reuse=False):
    inpdim = inp.get_shape().as_list()[3]
    lin_mu = convlayer(name+'_color', inp, ksize, inpdim, outdim, 1, reuse=reuse, nonlin=tf.nn.sigmoid, dobn=False, padding='SAME')
    mu = -0.1 + 1.2*lin_mu
    return mu


def batchnorm(X, is_training, reuse=False, decay=0.9, name='batchnorm'):
    assert len(X.get_shape().as_list())==4, 'input must be 4d tensor'

    outdim = X.get_shape().as_list()[-1]

    beta = tf.get_variable('beta_'+name, dtype=tf.float32, shape=outdim, initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable('gamma_'+name, dtype=tf.float32, shape=outdim, initializer=tf.constant_initializer(1.0))
    pop_mean = tf.get_variable('pop_mean_'+name, dtype=tf.float32, shape=outdim, trainable=False)
    pop_var = tf.get_variable('pop_var_'+name, dtype=tf.float32, shape=outdim, trainable=False)

    batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2], name='moments_'+name)

    if not reuse:
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            pop_mean_op = tf.assign(pop_mean, ema.average(batch_mean))
            pop_var_op = tf.assign(pop_var, ema.average(batch_var))

            with tf.control_dependencies([ema_apply_op, pop_mean_op, pop_var_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training, mean_var_with_update,
                    lambda: (pop_mean, pop_var))
    else:
        mean, var = tf.cond(is_training, lambda: (batch_mean, batch_var),
                lambda: (pop_mean, pop_var))

    return tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-5)



def train(loss, global_step, learning_rate, target_vars=None, name='', moving_average_decay=0.99):
    # Decay the learning rate exponentially based on the number of steps.
    opt = tf.train.AdamOptimizer(learning_rate)
    #opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.99, use_nesterov=True)
  
    if not target_vars:
        target_vars = tf.trainable_variables()
    grads = opt.compute_gradients(loss, var_list=target_vars)
  
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    ## Track the moving averages of all trainable variables.
    #variable_averages = tf.train.ExponentialMovingAverage(
    #    moving_average_decay, global_step)
    #if target_vars:
    #    variables_averages_op = variable_averages.apply(target_vars)
    #else:
    #    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    #with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    #    train_op = tf.no_op(name='train')
  
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name=name)
  
    return train_op


class GaussianParameterizer(object):

    def __init__(self, inpdim, outdim, name='', ksize=1, fcdim=256):
        self.inpdim = inpdim
        self.outdim = outdim
        self.ksize = ksize
        self.fcdim = fcdim
        self.name = name

    def get_params(self, inp, is_training, reuse=False):
        mu = convlayer(self.name+'_mu', inp, self.ksize, self.inpdim, self.outdim, stride=1, reuse=reuse, nonlin=tf.identity, dobn=False, padding='SAME', is_training=is_training)
        sigma = convlayer(self.name+'_sigma', inp, self.ksize, self.inpdim, self.outdim, stride=1, nonlin=tf.nn.softplus, dobn=False, reuse=reuse, padding='SAME', is_training=is_training)
        return mu, sigma


class GumbelSoftmaxParameterizer(object):

    def __init__(self, inpdim, outdim, name='', ksize=1):
        self.inpdim = inpdim
        self.outdim = outdim
        self.ksize = ksize
        self.name = name

    def get_params(self, inp, is_training, reuse=False):
        # logits
        logits_y = convlayer(self.name+'_logits', inp, self.ksize, self.inpdim, self.outdim, stride=1, reuse=reuse, nonlin=tf.identity, dobn=False, padding='SAME', is_training=is_training)
        return logits_y

def sample_gumbelsoftmax(logits, name, temperature, hard=False):
    def sample_gumbel(shape, eps=1e-20): 
        """Sample from Gumbel(0, 1)"""
        U = tf.random_uniform(shape,minval=0,maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)
    
    def gumbel_softmax_sample(logits, temperature): 
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + sample_gumbel(tf.shape(logits))
        return tf.nn.softmax( y / temperature)
    
    def gumbel_softmax(logits_y, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        batch_size = logits_y.get_shape().as_list()[0]
        h = logits_y.get_shape().as_list()[1]
        w = logits_y.get_shape().as_list()[2]
        category_num = 4
        outdim = logits_y.get_shape().as_list()[3]
        assert outdim//category_num==outdim/category_num, 'outdim(%d) must be exactly divisible by category_num(%d)' % (outdim, category_num)
        logits_y = tf.reshape(logits_y, [batch_size, h, w, outdim//category_num, category_num])
        y = gumbel_softmax_sample(logits_y, temperature)
        if hard:
            k = tf.shape(logits_y)[-1]
            #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
            y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
            
        y = tf.reshape(y, [batch_size, h, w, outdim])
        return y

    return gumbel_softmax(logits, temperature, hard)


def gumbelsoftmax_kldiv(logits_y):
    batch_size = logits_y.get_shape().as_list()[0]
    h = logits_y.get_shape().as_list()[1]
    w = logits_y.get_shape().as_list()[2]
    outdim = logits_y.get_shape().as_list()[3]
    logits_y = tf.reshape(logits_y, [batch_size, h, w, outdim//2, 2])
    q_y = tf.nn.softmax(logits_y) 
    log_q_y = tf.log(q_y + 1e-20)
    kl_tmp = q_y*(log_q_y-tf.log(1.0/2))
    KL = tf.reduce_sum(kl_tmp,[1, 2, 3, 4])
    return KL


def sample_gaussian(codes_mu, codes_sigma, noise, name, coeff=1.0):
    codes = codes_mu + coeff*(codes_sigma*noise)
    return codes


def kldiv_unitgauss(mu, sigma, coeff=+1.0, varname=None):
    kldiv = -0.5 * coeff * tf.reduce_sum(1 + 2.0*tf.log(sigma) - tf.square(mu) - tf.square(sigma), [1, 2, 3], name=varname)
    return kldiv 


