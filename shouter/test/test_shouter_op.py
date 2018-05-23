
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

def dump_variable(vars):
    print('\n'.join(["%s %s %s" %(v.name, v.shape, v.dtype) for v in vars]))


def dump_compute_operations(ops):
    # each varibable has a compute op
    def opstr(op):
        s  = "%s(%s)" % (op.name, op.type)
        if len(op.outputs) == 0:
            return s
        
        s += ', in:' + ','.join([t.name for t in op.inputs])
        s += ', dep:' + ','.join([o.name for o in op.control_inputs])
        return s
    
    print('\n'.join([opstr(op) for op in ops]))  

def assign_global_variables(graph):
    global_vals = tf.global_variables()
    dump_variable(global_vals)



## test graph topo 

if __name__ == '__main__':
    
    print(shouter.init())

    class_num    = 10
    weight_decay = 0.0004

    x = tf.ones([2, 3, 32, 32], tf.float32)
    z = tf.ones([2], tf.int32)
    
    lr  = tf.constant(0.1)
    opt = tf.train.MomentumOptimizer(lr,0.9)

    global_step   = tf.train.get_or_create_global_step()
    

    with tf.device('/GPU:0'):
        y,_ = model(x, class_num)
        loss = tf.losses.sparse_softmax_cross_entropy(logits=y, labels=z)
        loss = tf.reduce_mean(loss)
        
        trainable_variable = tf.trainable_variables()
        l2loss = [tf.nn.l2_loss(v) for v in trainable_variable]
        l2loss = weight_decay * tf.add_n(l2loss)

        loss = loss + l2loss

        gradvars = opt.compute_gradients(loss=loss, var_list=trainable_variable)
        reduce_g = []
        for g, v in gradvars:
            reduceg = shouter._allreduce(g)
            reduce_g.append((reduceg, v))
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        apply_ops = opt.apply_gradients(reduce_g)
    

    
    update_global = global_step.assign_add(1)
    train_op=[update_ops, apply_ops, global_step]
 
    
    print(global_step)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4

    # start train loop 
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        #assign_global_variables(sess.graph)
        #dump_compute_operations(sess.graph.get_operations())

        yp = sess.run(train_op)
        #print(yp)
 

