#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# keep easy understand and simple to extend.
# same as keras https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py 

import tensorflow as tf
import yaml
import sys 

"""
As keep simple and keep scalability, just wrapper the min logic
this just a configure file
"""

tf.logging.set_verbosity(tf.logging.DEBUG)

# modules
sys.path.append('.')

from name_dict import NameDict

from data_set      import cifar10_dataset
from model         import official_model
from learning_rate import PiecewiseLR
from pipeline      import pipeline

'''
usage: 
   cifar10_train.py [cifar10_train.yaml]
   
'''


def update_param(pipe):
    train_file = './cifar10_train.yaml' if len(sys.argv) == 1 else sys.argv[1]

    print(train_file)
    with open(train_file, 'r') as f:
        param = yaml.load(f)

    param['all_step'] = max(int(param.epoch * param.train_nums / param.minibatch), 10)
    param['batch_size'] = max(int(param.minibatch / pipe.gpu_nums), 1)
    param['minibatch'] = param.batch_size*pipe.gpu_nums

    #this is nessary compute lr with epoch 
    param['train_nums'] = (param.train_nums // param.minibatch) * param.minibatch

    tf.logging.info("Param:" + repr(param))
    return param

def main(argv):
    pipe  = pipeline.Pipeline()

    # 1. config load, as hypter params should define in yaml config
    param = update_param(pipe) 

    # 2. selet a model, dataset, lr, opt, and so on,  as these can be enumeration.
    create_model_func = official_model.ResnetModel(param.resnet_layer, param.class_num)

    train_set, vaild_set, eval__set = cifar10_dataset(param.data_path, param.batch_size, param.epoch)

    lr = PiecewiseLR(param)
    opt = tf.train.MomentumOptimizer(lr, param.momentum)

    #3 set up graph these should not modify
    #3.1 set_up dataset
    device_dataset = pipe.setup_dataset(train_set)

    #3.2 set_up model and loss    
    device_labels, device_predicts = pipe.setup_model(device_dataset, create_model_func)
    
    
    def compute_loss_func(labels, logits):
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        return tf.reduce_mean(loss)
    device_losses, sum_loss = pipe.setup_loss(device_labels, device_predicts, compute_loss_func, param.weight_decay)
    
    def accuracy(labels, predicts, topk):
        return tf.reduce_sum(tf.cast(tf.nn.in_top_k(predicts, labels, topk), tf.float32))

    top1 = pipe.setup_reducesum(device_labels, device_predicts, lambda x,y:accuracy(x,y, 1))
    top5 = pipe.setup_reducesum(device_labels, device_predicts, lambda x,y:accuracy(x,y, 5))

    
    #3.3 set_up gradient compute and update
    train_op = pipe.setup_train(device_losses, opt)
    
    global_step = tf.train.get_global_step()
    #TODO add restore hook
    
    hooks = pipe.get_hook() + [
         tf.train.StopAtStepHook(last_step = param.all_step),
         tf.train.LoggingTensorHook(tensors={'step': global_step,
                                             'learning_rate:': lr,
                                             'sum_loss': sum_loss,
                                             'batch_top1': top1,
                                             'batch_top5': top5,
                                             },
                                   every_n_iter=10),
    ]
    

    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4

    
    # start train loop 
    with tf.train.MonitoredTrainingSession(checkpoint_dir=param.checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run([train_op])
            
if __name__ == "__main__":
    tf.app.run()
