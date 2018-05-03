import tensorflow as tf
import itertools

from tensorflow.python.framework import tensor_util




class TestData:
    def __init__(self, shape, classnum):
        '''
         image_shape=(CHW)
         labelnum
       '''
        self.batch_size, self.channel, self.height, self.width = shape
        self.classnum = classnum

        images = tf.truncated_normal(
            [self.channel, self.height, self.width],
            mean=127,
            stddev=120,
            dtype=tf.float32,
            seed=0,
            name='synthetic_images')

        labels = tf.random_uniform(
            [1],
            minval=0,
            maxval=classnum - 1,
            dtype=tf.int32,
            name='synthetic_labels')
        self.dataset = tf.data.Dataset.from_tensors((images, labels))

    def get_next(self, workernum, workerdevice='GPU'):

        ds = self.dataset
        batch_size = self.batch_size

        # we can ajust this by cpu memsize
        shuffle_buffer_size = batch_size * workernum * 100

        ds.prefetch(shuffle_buffer_size * 2)
        ds = ds.apply(
            tf.contrib.data.shuffle_and_repeat(shuffle_buffer_size, -1))

        def parser(image, label):

            #image = tf.cast(image, tf.float32)
            image = image / 128 - 1
            image = tf.image.random_flip_left_right(image)

            label = tf.cast(label, tf.int32)
            return image, label

        num_parallel_batches = max(batch_size >> 2, 1)
        ds = ds.apply(
            tf.contrib.data.map_and_batch(
                map_func=parser,
                batch_size=batch_size,
                num_parallel_batches=num_parallel_batches))

        iterator = ds.make_initializable_iterator()

        image_batch, label_batch = iterator.get_next()

        enqueue_ops = []
        output_ops = {}

        for i in range(workernum):
            with tf.device('/device:%s:%d' % (workerdevice, i)):
                gpu_stage = tf.contrib.staging.StagingArea(
                    dtypes=[image_batch.dtype, label_batch.dtype],
                    shapes=[image_batch.get_shape(),
                            label_batch.get_shape()])

                put_gpu_op = gpu_stage.put([image_batch, label_batch])
                gpu_output = gpu_stage.get()
                gpu_size   =  gpu_stage.size()

                # add debug
                gpu_output[0] = tf.Print(gpu_output[0], [gpu_stage.size()], "gpu_stage size:");

                enqueue_ops.append(put_gpu_op)
                output_ops[i] = gpu_output

        return iterator, enqueue_ops, output_ops

# test 
if __name__ == '__main__':
    
    
    ds = TestData(shape=(2, 1, 1, 1), classnum=2)
    iter = ds.dataset.make_initializable_iterator()
    value = iter.get_next()

    sess = tf.Session()
    sess.run(iter.initializer)
    v = sess.run(value)  # (1, array([1]))
    print(v)
