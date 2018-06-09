#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

'''
just for easy dataset import

'''
import os
import tensorflow as tf
from  augment import imagenet_preprocessing

def synthetic_dataset(shape, classnum):
    """create synthesis data for classify test.

    Args:
      shape: [n,c,w,h].
      classnum: int, must be ranage[0, INT_MAX)

    Return :
      dataset: tf.data.Dataset instance
    """
    batch_size, channel, height, width = shape

    images = tf.truncated_normal(
        [batch_size, channel, height, width],
        mean=127,
        stddev=120,
        dtype=tf.float32,
        seed=0,
        name='synthetic_images')
    labels = tf.random_uniform(
        [batch_size],
        minval=0,
        maxval=classnum - 1,
        dtype=tf.int32,
        name='synthetic_labels')



    def parser(image, label):
        #image = tf.cast(image, tf.float32)
        image = image / 128.0 - 1.0
        #image = tf.image.random_flip_left_right(image)

        label = tf.cast(label, tf.int32)
        return image, label

    # as has one tensor so batch not work
    ds = tf.data.Dataset.from_tensors((images, labels))
    num_parallel = max(batch_size >> 2, 1)
    ds = ds.map(map_func=parser, num_parallel_calls=num_parallel)
    ds = ds.repeat(-1)

    return ds


def cifar10_dataset(path, batch_size, repeat=-1):
    """create synthesis data for classify.
       See http://www.cs.toronto.edu/~kriz/cifar.html.

    Args:
      path: the tfrecord path, which can be genrate by `genrate_cifar10_tfrecords.py`
      classnum: int, must be ranage[0, INT_MAX)

    Return :
      train_set, vaild_set, eval__set:
    """

    def parser(serialized_example):
        """cpoy for tensorflow tutorials."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # all this done cpu
        image = tf.decode_raw(features['image'], tf.uint8)
        # hard code
        image.set_shape([3 * 32 * 32])
        image = tf.reshape(image, [3, 32, 32])

        #output NHWC
        image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

        label = tf.cast(features['label'], tf.int32)
        
        return image, label

    def augment(image):
        # Reshape to CHW -> HWC for augment
        image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
        image = tf.random_crop(
            image, [32, 32, 3])  # size = [crop_height, crop_width, 3].

        image = tf.image.random_flip_left_right(image)

        #"32,32,3  -> 3,32,32 HWC -> CHW"
        #image = tf.transpose(image, [2, 0, 1])
        #Todo show image
        return image

    def nomarlize(image):
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1.0
        #image = tf.image.per_image_standardization(image)

        return image

    def load_tfrecords(filenames, batch_size, repeat=-1, istrain=True):

        num_parallel_batches = max(batch_size >> 2, 1)

        def map_fun(x):
            image, label = parser(x)
            image = augment(image) if istrain else image
            image = nomarlize(image)
            return (image, label)

        #ds = tf.data.Dataset.from_generator((f for f in filenames),
        #                                    (tf.string))
        ds = tf.data.Dataset.from_tensors(filenames)
        ds = ds.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=len(filenames)))
        ds = ds.prefetch(batch_size * 2)

        ds = ds.apply(
            tf.contrib.data.shuffle_and_repeat(batch_size * 2, repeat))

        ds = ds.apply(
            tf.contrib.data.map_and_batch(
                map_func=map_fun,
                batch_size=batch_size,
                num_parallel_batches=num_parallel_batches,
                drop_remainder=True))
        
        ds = ds.prefetch(batch_size)
        return ds
    
    train_set = load_tfrecords(
        path + "/train.tfrecords", batch_size, repeat=-1, istrain=True)
    vaild_set = load_tfrecords(
        path + "/validation.tfrecords", batch_size, repeat=1, istrain=False)
    eval__set = load_tfrecords(
        path + "/eval.tfrecords", batch_size, repeat=1, istrain=False)

    return train_set, vaild_set, eval__set


def imagenet_dataset(path, batch_size):
    """create imagenet data for classify.
        See https://github.com/tensorflow/models/tree/master/research/slim/datasets
        See also: https://github.com/tensorflow/models/offical/resnet

    Args:
      path: the tfrecord path
      classnum: int, must be ranage[0, INT_MAX)

    Return :
      train_set, vaild_set:
    """
    if not os.path.exists(path):
        raise FileExistsError(path)

    def parser(serialized_example):
        """cpoy for tensorflow benchmarks.

        image/height: 462
        image/width: 581
        image/colorspace: 'RGB'
        image/channels: 3
        image/class/label: 615
        image/class/synset: 'n03623198'
        image/class/text: 'knee pad'
        image/object/bbox/xmin: 0.1
        image/object/bbox/xmax: 0.9
        image/object/bbox/ymin: 0.2
        image/object/bbox/ymax: 0.6
        image/object/bbox/label: 615
        image/format: 'JPEG'
        image/filename: 'ILSVRC2012_val_00041207.JPEG'
        image/encoded: <JPEG encoded string>

        """
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/encoded':
                tf.FixedLenFeature([], tf.string, default_value=''),
                'image/class/label':
                tf.FixedLenFeature([], tf.int64, default_value=-1),
                'image/object/bbox/xmin':
                tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin':
                tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax':
                tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax':
                tf.VarLenFeature(dtype=tf.float32)
            })

        label = tf.cast(features['image/class/label'], tf.int32)
        image = features['image/encoded']

        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

        # Note that we impose an ordering of (y, x) just to make life difficult.
        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

        # Force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        # [1, coords, num_boxes]. ??
        bbox = tf.transpose(bbox, [0, 2, 1])

        return image, label, bbox

    def augment(image, bbox, istrain):
        image = imagenet_preprocessing.preprocess_image(image, bbox, 224, 224, 3, istrain)
        return image


    def load_tfrecords(pattern, batch_size, repeat=-1, istrain=True):

        num_parallel_batches = 1

        def map_fun(x):
            image, label, bbox = parser(x)
            image = augment(image, bbox, istrain)
             
            """ image has shape [224, 224, 3] need to transpos NCHW"""
            #if istrain:
            #    image = tf.transpose(image, [2, 0, 1]) 
            return (image, label)
        
        # this is bug // can hold the input
        pattern = '/'.join([p for p in pattern.split('/') if len(p) > 0])
        ds = tf.data.Dataset.list_files(pattern)
        ds = ds.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=20))

        ds = ds.prefetch(batch_size)

        ds = ds.apply(
              tf.contrib.data.shuffle_and_repeat(batch_size * 100, repeat))

        ds = ds.apply(
              tf.contrib.data.map_and_batch(
                map_func=map_fun,
                batch_size=batch_size,
                num_parallel_batches=num_parallel_batches,
                drop_remainder=True))

        ds = ds.prefetch(batch_size)
        return ds

    train_set = load_tfrecords(
        path + "/train*", batch_size, repeat=-1, istrain=True)
    vaild_set = load_tfrecords(
        path + "/validation*", batch_size, repeat=1, istrain=False)
    return train_set, vaild_set


if __name__ == '__main__':

    '''
    ds = synthetic_dataset([4, 1, 1, 1], classnum = 10)
    iterator = ds.make_initializable_iterator()
    image, label = iterator.get_next()
    
    ops = tf.get_default_graph().get_operations()
    print(ops)
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        print("Test synthetic_dataset")
        print(sess.run([image, label]))

    '''
    '''
    train_set, vaild_set, eval__set = cifar10_dataset('./test_dataset/cifar10', batch_size=2, repeat=-1)
    iterator = train_set.make_initializable_iterator()
    image, label = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        print("Test cifar10_dataset")
        print(sess.run([image, label]))
    '''
    
    train_set, vaild_set = imagenet_dataset('./test_dataset/imagenet_sample', batch_size=2)
    train_iterator = train_set.make_initializable_iterator()
    vaild_iterator = vaild_set.make_initializable_iterator()

    train_image, train_label = train_iterator.get_next()
    vaild_image, vaild_label = vaild_iterator.get_next()

    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    scaffold = tf.train.Scaffold()

    with tf.train.MonitoredTrainingSession(hooks=[],  scaffold = scaffold, config=config) as sess:
        sess.run(train_iterator.initializer)
        sess.run(vaild_iterator.initializer)
        #print(sess.run([image, label, bbox]))
        for i in range(0,3):
            v1, v2 = sess.run([train_image, vaild_image])
            print('Test imagenet %s %s' % (v1.shape, v2.shape))
    

