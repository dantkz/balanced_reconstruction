from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

import scipy.ndimage
import glob
import os
import multiprocessing



def read_image(fname):
    image = np.array(scipy.ndimage.imread(fname))
    image = image.astype(np.float32)/255.0
    return image


class FolderDataset(object):

    def __init__(self, path, num_img=None, ind_offset=0, file_extension='png'):
        """Construct a Dataset.
        """
        self.ind_offset = int(ind_offset)
        print(path)
        print(os.path.expanduser(path))
        print(os.path.realpath(os.path.expanduser(path)))

        path = os.path.realpath(os.path.expanduser(path))

        print('reading', path)
        filelist = sorted(glob.glob(os.path.join(path, '*.' + file_extension)))
        
        def get_dims(fname):
            image = np.array(scipy.ndimage.imread(fname))
            return image.shape[0], image.shape[1], image.shape[2]

        print(len(filelist))
        self.height, self.width, self.color_chn = get_dims(filelist[0])

        self._index_in_epoch = 0
        num_examples = len(filelist)
        if num_img==None:
            self.num_img = num_examples - self.ind_offset
        else:
            self.num_img = num_img
            assert num_img <= num_examples - self.ind_offset

        self.tffilelist = tf.convert_to_tensor(filelist)

        print('File index offset: ' + str(self.ind_offset))
        print('Number of examples: ' + str(num_examples))
        print('Number of examples to train: ' + str(self.num_img))

    def get_dims(self):
        return self.height, self.color_chn

    def get_image(self, filenames):
        reader = tf.WholeFileReader()
        k, val = reader.read(filenames)
        image = tf.image.decode_png(val)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image.set_shape([self.height, self.width, self.color_chn])
        return image

    def next_batch(self, batch_size, doperm=True):
        filenames = tf.train.string_input_producer(self.tffilelist, shuffle=doperm, name='filenames_producer')
        image = self.get_image(filenames)
        min_after_dequeue = 2048
        capacity = min_after_dequeue + 4 * batch_size
        images = tf.train.shuffle_batch([image],
            batch_size=batch_size, min_after_dequeue=min_after_dequeue, capacity=capacity, num_threads=4)
        return images


def get_dataset(dataset_name):
    if dataset_name=='leaves':
        folder_path = 'leaves/scaled/'
        num_img = 628
        ind_offset = 0
    elif dataset_name=='celeba64':
        #folder_path = '~/scratch/Datasets/CelebA/mycrop64/'
        folder_path = "~/Documents/Datasets/CelebA/mycrop64/"
        num_img = 190000
        ind_offset = 0
    return get_folder(folder_path, 
                num_img=num_img, 
                ind_offset=ind_offset)


def get_folder(path, num_img=None, ind_offset=None):
    if ind_offset is None:
        ind_offset = 0

    train = FolderDataset(path, num_img, ind_offset)
    #test = FolderDataset(path, None, num_img)

    class Datasets(object):
        pass

    result = Datasets()
    result.train = train
    #result.test = test
    return result



def test():
    folder_path = "~/Documents/Datasets/CelebA/mycrop64/"
    num_img = 200
    ind_offset = 0
    
    data = get_folder(folder_path, 
                num_img=num_img, 
                ind_offset=ind_offset)

    train_images = data.train.next_batch(4, doperm=True)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())
        images = sess.run(train_images)
        print(images.shape)

        coord.request_stop()
        coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
    test()

if __name__ == '__main__':
    tf.app.run()
