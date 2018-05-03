import tensorflow as tf
import itertools

from tensorflow.python.framework import tensor_util

class Model:
    def __init__(self, isTrain=True):
        self._is_training = isTrain
        self._data_format = 'channels_first'
        self._batch_norm_decay = 0.997
        self._batch_norm_epsilon = 1e-5
        self.weight_decay = 0.001

    def _bottleneck_residual_v2(self,
                                x,
                                in_filter,
                                out_filter,
                                stride,
                                activate_before_residual=False):
        """Bottleneck residual unit with 3 sub layers, plan B shortcut."""

        with tf.name_scope('bottle_residual_v2') as name_scope:
            if activate_before_residual:
                x = self._batch_norm(x)
                x = self._relu(x)
                orig_x = x
            else:
                orig_x = x
                x = self._batch_norm(x)
                x = self._relu(x)

            x = self._conv(x, 1, out_filter // 4, stride, is_atrous=True)

            x = self._batch_norm(x)
            x = self._relu(x)
            # pad when stride isn't unit
            x = self._conv(x, 3, out_filter // 4, 1, is_atrous=True)

            x = self._batch_norm(x)
            x = self._relu(x)
            x = self._conv(x, 1, out_filter, 1, is_atrous=True)

            if in_filter != out_filter:
                orig_x = self._conv(
                    orig_x, 1, out_filter, stride, is_atrous=True)
            x = tf.add(x, orig_x)

            tf.logging.info('image after unit %s: %s', name_scope,
                            x.get_shape())
            return x

    def _conv(self, x, kernel_size, filters, strides, is_atrous=False):
        """Convolution."""

        padding = 'SAME'
        if not is_atrous and strides > 1:
            pad = kernel_size - 1
            pad_beg = pad // 2
            pad_end = pad - pad_beg
            if self._data_format == 'channels_first':
                x = tf.pad(
                    x,
                    [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
            else:
                x = tf.pad(
                    x,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            padding = 'VALID'
        return tf.layers.conv2d(
            inputs=x,
            kernel_size=kernel_size,
            filters=filters,
            strides=strides,
            padding=padding,
            use_bias=False,
            data_format=self._data_format)

    def _batch_norm(self, x):
        if self._data_format == 'channels_first':
            data_format = 'NCHW'
        else:
            data_format = 'NHWC'
        return tf.contrib.layers.batch_norm(
            x,
            decay=self._batch_norm_decay,
            center=True,
            scale=True,
            epsilon=self._batch_norm_epsilon,
            is_training=self._is_training,
            fused=True,
            data_format=data_format)

    def _relu(self, x):
        return tf.nn.relu(x)

    def _fully_connected(self, x, out_dim):
        with tf.name_scope('fully_connected') as name_scope:
            x = tf.layers.dense(x, out_dim)

        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _avg_pool(self, x, pool_size, stride):
        with tf.name_scope('avg_pool') as name_scope:
            x = tf.layers.average_pooling2d(
                x, pool_size, stride, 'SAME', data_format=self._data_format)

        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _global_avg_pool(self, x):
        with tf.name_scope('global_avg_pool') as name_scope:
            assert x.get_shape().ndims == 4
            if self._data_format == 'channels_first':
                x = tf.reduce_mean(x, [2, 3])
            else:
                x = tf.reduce_mean(x, [1, 2])
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def inference(self, image, classnum):
        
        x = image
    
        x = self._bottleneck_residual_v2(x, 0, 16, 1, False)
        x = self._global_avg_pool(x)
        x = self._fully_connected(x, classnum)
        return x

    def loss(self, logits, label, opt):

        tower_pred = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabitlities': tf.nn.softmax(logits)
        }

        tower_loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=label)
        tower_loss = tf.reduce_mean(tower_loss)

        model_params = tf.trainable_variables()
        
        

        tower_loss += self.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in model_params])

        #tower_grad = tf.gradients(tower_loss, model_params)
        #grad_vars = list(zip(tower_grad, model_params))
        
        grad_vars = opt.compute_gradients(loss=tower_loss, var_list=model_params)
        
   
        
        #print(model_params)
        #print(tower_grad)
        
        return tower_loss, grad_vars , tower_pred
        
if __name__ == '__main__':
    from d import TestData
    
    ds = TestData(shape=(2, 1, 1, 1), classnum=2)
    iterator, enqueue_ops, output_ops = ds.get_next(2)
    m = Model()
    
    with tf.device('/GPU:0'):
        image, label = output_ops[0]
        out = m.inference(image, 10)
        tower_loss, grad_vars , tower_pred = m.loss(out, label)
        
 
    with tf.Session() as sess:
       sess.run(iterator.initializer)
       print(tf.global_variables_initializer())
       sess.run(tf.global_variables_initializer())
       
       for i in range(10):
          r = sess.run(enqueue_ops)  # (1, array([1]))
          gops = [g for g,v in grad_vars]
          rr= sess.run(gops)
          print(*zip(gops,rr))
