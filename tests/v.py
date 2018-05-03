# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import itertools
import six
import logging
import sys

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2

from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter

FORMAT = '%(asctime)-15s %(levelname)s %(process)d-%(thread)d %(message)s %(filename)s:%(module)s:%(funcName)s:%(lineno)d'
#logging.basicConfig(format=FORMAT, level=logging.DEBUG, handlers=[logging.StreamHandler(),logging.FileHandler(filename=sys.argv[0]+".log", mode='w')])
tf.logging.set_verbosity(tf.logging.INFO)

## now we only focus mutli gpu
# DDR4 2400：19.2 GB/s
#

from d import TestData
from m import Model
from h import ExamplesPerSecondHook

from graphviz_visual import print_tensortree


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
  if ps_ops == None:
    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

  if ps_strategy is None:
    ps_strategy = device_setter._RoundRobinStrategy(num_devices)
  if not six.callable(ps_strategy):
    raise TypeError("ps_strategy must be callable")

  def _local_device_chooser(op):
    current_device = pydev.DeviceSpec.from_string(op.device or "")

    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    if node_def.op in ps_ops:
      ps_device_spec = pydev.DeviceSpec.from_string(
          '/{}:{}'.format(ps_device_type, ps_strategy(op)))

      ps_device_spec.merge_from(current_device)
      return ps_device_spec.to_string()
    else:
      worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
      worker_device_spec.merge_from(current_device)
      return worker_device_spec.to_string()
  return _local_device_chooser


# now we has three method for multi-gpu sync
# 1 var placement cpu, each GPU has a copy, just like ps, worker
# 2 var placement a single GPU, other read or update
# 3 each gpu has a copy, use nccl all ring reduce


def scanfolld_init_fn(scaffold, session):
    #session.run(scaffold.my_init_ops)
    tf.logging.info("scanfolld_init_fn done------------------")
    print("scanfolld_init_fn done------------------")



def train():
    classnum = 10
    gpu_num = 1
        
    learning_rate = 1
    train_batch_size = 64
    eval_batch_size = 100

    num_batches_per_epoch = 32
    train_steps = 100

    num_workers = 1
    momentum = 0.9

    batch_norm_decay = 0.997
    batch_norm_epsilon = 1e-5


    ds = TestData(shape=(2, 3, 40, 40), classnum=classnum)

    starter, enqueue_ops, output_ops_map = ds.get_next(gpu_num)
    output_ops = list(output_ops_map.values())


    model = Model()

    tower_losses = []
    tower_gradvars = []
    tower_preds = []
    labels = []
    
    global_step = tf.train.get_or_create_global_step()

    boundaries = [
            num_batches_per_epoch * x
            for x in np.array([82, 123, 300], dtype=np.int64)
        ]
    staged_lr = [learning_rate * x for x in [1.0, 0.1, 0.01, 0.002]] 
        

    # why implement by c++ 
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, staged_lr)

    optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum)
        
    
    optimizer = tf.train.SyncReplicasOptimizer(
                optimizer, replicas_to_aggregate=num_workers)
    sync_replicas_hook = optimizer.make_session_run_hook(True)
    



    for i in range(gpu_num):
        worker_device = '/{}:{}'.format('GPU', i)

        device_setter = local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                gpu_num, tf.contrib.training.byte_size_load_fn))

        with tf.variable_scope('resnet', reuse=bool(i != 0)):
            with tf.name_scope('tower_%d' % i) as name_scope:
                with tf.device(device_setter):
                    imagebatch, labelbatch = output_ops_map[i]
                    out = model.inference(imagebatch, classnum)
                    loss, gradvars, preds = model.loss(out, labelbatch, optimizer)

                    print_tensortree(out)
                    labels.append(labelbatch)
                    tower_losses.append(loss)
                    tower_gradvars.append(gradvars)
                    tower_preds.append(preds)
                    if i == 0:
                        # Only trigger batch_norm moving mean and variance update from
                        # the 1st tower. Ideally, we should grab the updates from all
                        # towers but these stats accumulate extremely fast so we can
                        # ignore the other stats from the other towers without
                        # significant detriment.
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                       name_scope)

    #print_tensortree(out)
    gradvars = []
    with tf.name_scope('gradient_averaging'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_gradvars):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)
       
        #print(all_grads)
        for var, grads in six.iteritems(all_grads):
            # Average gradients on the same device as the variables
            # to which they apply.
            #
            # print((var.device, var))
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                    
            gradvars.append((avg_grad, var))
    
    # Device that runs the ops to apply global gradient updates.
    consolidation_device = '/device:GPU:0'
    with tf.device(consolidation_device):
        # Suggested learning rate scheduling from
        # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
        
       
        loss = tf.reduce_mean(tower_losses, name='loss')

        examples_sec_hook = ExamplesPerSecondHook(
            train_batch_size, every_n_steps=10, initer=starter.initializer, enqs=enqueue_ops)

        tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}
        #tensors_to_log = {'learning_rate': learning_rate}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)        

        # Create single grouped train op

        apply_grad = optimizer.apply_gradients(
                gradvars, global_step=tf.train.get_global_step())
        train_op = enqueue_ops + [ apply_grad , update_ops ]
        train_op = tf.group(*train_op)

        runner = optimizer.get_chief_queue_runner()
        #print(tower_preds)

        predictions = {
            'classes':
            tf.concat([p['classes'] for p in tower_preds], axis=0),
            'probabitlities':
            tf.concat([p['probabitlities'] for p in tower_preds], axis=0)
        }
        stacked_labels = tf.concat(labels, axis=0)
        metrics = {
            'accuracy': tf.metrics.accuracy(stacked_labels,
                                            predictions['classes'])
        }

    #init_ops = tf.group(tf.global_variables_initializer(), iterator.initializer)
    # see https://www.tensorflow.org/api_docs/python/tf/train/Scaffold
    
    print("GLOBAL_VARIABLES:" + repr(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
    print("LOCAL_VARIABLES:" + repr(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)))
    print("METRIC_VARIABLES:" + repr(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)))
    print("MODEL_VARIABLES:" + repr(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)))
    
    print("TRAINABLE_VARIABLES:" + repr(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    print("MOVING_AVERAGE_VARIABLES:" + repr(tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)))
    print("CONCATENATED_VARIABLES:" + repr(tf.get_collection(tf.GraphKeys.CONCATENATED_VARIABLES)))
    print("TRAINABLE_RESOURCE_VARIABLES:" + repr(tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)))

    
    print("QUEUE_RUNNERS:" + repr(tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)))
    print("REGULARIZATION_LOSSES:" + repr(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    # just init variable
    scaffold = tf.train.Scaffold(
        init_fn=scanfolld_init_fn)


    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    print("gpu_options:"+repr(config.gpu_options))

    checkpoint_dir = './checkpoint'
    train_hooks = [logging_hook, examples_sec_hook, sync_replicas_hook]

    #train_hooks = [examples_sec_hook, logging_hook]


    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # see https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession
    with tf.train.MonitoredTrainingSession(
            is_chief=True,
            checkpoint_dir=checkpoint_dir,
            scaffold=scaffold,
            hooks=train_hooks,
            save_checkpoint_secs=-1,
            config=config) as mon_sess:
       
        tf.train.start_queue_runners(mon_sess, runner)
        #init 
        #mon_sess.run(starter.initializer)
        i = 0 
        while not mon_sess.should_stop() and  i < 100:            
            mon_sess.run(train_op, options=run_options, run_metadata=run_metadata)            


 




if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
    os.system("rm ./checkpoint/*")
    train()
