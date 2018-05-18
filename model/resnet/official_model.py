import tensorflow as tf

from . import resnet_model

class ImageNetModel:
    '''
    work for imagenet: copy from https://github.com/tensorflow/models/tree/master/official
    '''
    default_layer_config = {
        18: [2, 2, 2, 2],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    def __init__(self, layernums=50, num_classes=10000):
        self.data_format = 'channels_first'
        self.version = resnet_model.DEFAULT_VERSION  # use v2
        self.dtype = tf.float32
        self.bottleneck, self.final_size = (False, 512) if layernums < 50 else (True, 2048)

        self.model = resnet_model.Model(
            resnet_size=layernums,
            bottleneck=self.bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            second_pool_size=7,  #no used
            second_pool_stride=1,  #no used
            block_sizes=ImageNetModel.default_layer_config[layernums],
            block_strides=[1, 2, 2, 2],
            final_size=self.final_size,
            version=self.version,
            data_format=self.data_format)

    def __call__(self, inputs, training=True):
        '''
        inputs should be NCHW
        '''
        return self.model(inputs, training)

class Cifar10Model:

    def __init__(self, layernums=32, num_classes=10):
        self.data_format = 'channels_first'
        self.version = resnet_model.DEFAULT_VERSION  # use v2
        self.dtype = tf.float32
        self.final_size = 64

        if layernums % 6 != 2:
          raise ValueError('resnet_size must be 6n + 2:', resnet_size)

        num_blocks = (layernums - 2) // 6

        self.model = resnet_model.Model(
            resnet_size=layernums,
            bottleneck=False,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            second_pool_size=8,  #no used
            second_pool_stride=1,  #no used
            block_sizes=[num_blocks] * 3,
            block_strides=[1, 2, 2, 2],
            final_size=self.final_size,
            version=self.version,
            data_format=self.data_format)

    def __call__(self, inputs, training=True):
        return self.model(inputs, training)
        

if __name__ == "__main__":

    import sys
    sys.path.append('.')

    from tools.graphviz_visual import print_tensortree

    m = ResnetModel(34, 10)
    g = tf.Graph()
    with g.as_default():
        x = tf.ones([2, 3, 32, 32], tf.float32)
        x = m(x, training=False)
        print_tensortree(x)
    
        #sess = tf.Session()
        #sess.run(tf.global_variables_initializer())

        #v = sess.run(x)
        #print(v)
