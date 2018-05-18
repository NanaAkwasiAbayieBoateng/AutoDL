
import tensorflow as tf  

import sys

sys.path.append('build/lib.linux-x86_64-3.5')
from shouter import tensorflow as shouter

sys.path.append('..')
from model import mobilenet_v2


def model(x, class_num):
    m = mobilenet_v2.MobilenetModel()
    x = m.build_network(x, phase_train=True, nclass=class_num)
    return x
    

## test graph topo 

if __name__ == '__main__':
    
    print(shouter.init())

    class_num = 10
    x = tf.ones([2, 3, 32, 32], tf.float32)
    z = tf.ones([2, 10], tf.float32)
    
    with tf.device('/GPU:0'):
        y,_ = model(x, class_num)

    print(y)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        yp = sess.run(y)
        print(yp)



