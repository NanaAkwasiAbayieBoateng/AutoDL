'''
Test custom operation,
see also  zmq_ops
this demo a op converat a buffer[int32*4] to a tensorshape

Note:
The body of the function (i.e. func) will not be serialized in a GraphDef. 
Therefore, you should not use this function 
if you need to serialize your model and restore it in a different environment

'''

import struct
import os
import numpy as np

import tensorflow as tf

from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework import types_pb2 as DT

'''
'''

def struct_pack(lists):
    shape = tf.TensorShape(lists)
    return struct.pack('IIII', *shape.as_list())

def struct_unpack(bytes):    
    lists = struct.unpack('IIII', bytes)
    shape = tf.TensorShape(lists)    
    return np.array(shape.as_list(), np.int32)


def convert_shape_to_tensor(shape):
    return tf.convert_to_tensor(shape.as_list(), dtype=tf.int32)


tf.register_tensor_conversion_function(tf.TensorShape, convert_shape_to_tensor)
#@tf.RegisterGradient("PyFunc") 


def test_covert():
    shape = tf.TensorShape([1,2,3,4])
    tensor = tf.convert_to_tensor(shape)
    with tf.Session() as sess:
       v = sess.run(tensor)
       print(v)

def test_pack():
    x = tf.placeholder(tf.int32, shape=(4,))
    shapepack = tf.py_func(struct_pack, [x],tf.string)
    with tf.Session() as sess:
       v = sess.run(shapepack, feed_dict={x:[1,2,3,4]})
       print((type(v),v))
       
def test_unpack():
    z = tf.placeholder(tf.string, shape=())
    shapeunpack = tf.py_func(struct_unpack, [z], tf.int32)
    
    v = b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00'
    with tf.Session() as sess:
       v = sess.run([shapeunpack], feed_dict={z:v})
       print(v)



if __name__ == '__main__':
   print('test_covert')
   test_covert()
   
   print('test_pack')
   test_pack()
   
   print('test_unpack')
   test_unpack()
   

   
