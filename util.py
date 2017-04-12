from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
import math
import scipy.misc

from six.moves import xrange  # pylint: disable=redefined-builtin


def save_img(img, fname):
    if img.shape[-1] == 1:
        img = np.squeeze(img)
    scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(fname)


def image_summary(images, prefix=''):
    for j in xrange(min(10, images.get_shape().as_list()[0])):
        desc_str = '%d'%(j) + '_' + prefix
        tf.summary.image(desc_str, images[j:(j+1), :, :, :], max_outputs=1)

        
def activation_summary(x, tensor_name=None):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    #Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    if tensor_name is None:
        tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    #tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


