#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# keep easy understand and simple to extend.
# same as keras https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py 

import yaml
import sys
import os
import logging

from logging.handlers import TimedRotatingFileHandler


"""
As keep simple and keep scalability, just wrapper the min logic
this just a configure file
"""
#class TraceHandker(TimedRotatingFileHandler):
#TraceHandker(filename='a.log', when='h', interval=1)
class TraceHandker(logging.StreamHandler):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def emit(self, record):
        # 7 is get by debug
        frame = sys._getframe(7)
        code = frame.f_code
        record.msg = record.msg + " " + ":".join((code.co_name, code.co_filename,str(frame.f_lineno),))
        super().emit(record)
        
FORMAT = '%(asctime)-15s %(process)d %(levelname)s %(message)s'
handlers=[TraceHandker()]

logging.basicConfig(format=FORMAT, level=logging.DEBUG, handlers = handlers)

# modules
sys.path.append('.')

from name_dict import NameDict

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

from data_set      import cifar10_dataset
from model         import official_model
from learning_rate import PiecewiseLR
from pipeline      import multipipeline
from initialize    import InitScaffold
from evaluater      import Evaluater
import train_hook

'''
setup:
    1. install nccl2: 
        perfer : NCCL 2.1.0, this issue is no longer present when using CUDA 9,
        https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html
  
    2. install openmpi3.0:
        wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz && \
        tar zxf openmpi-3.0.0.tar.gz && \
        cd openmpi-3.0.0 && \
       ./configure --enable-orterun-prefix-by-default && \
        make -j $(nproc) all && \
        make install && \
        ldconfig && \
   
    3. install horovod
        sudo -E HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir -U horovod

    4. some configure:
        # Configure OpenMPI to run good defaults:
        # --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
        echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
        echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
        echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf
    
    5. # Set default NCCL parameters
        echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
        echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf
    6. for nvidia docker 
        --shm-size=1g --ulimit memlock=-1

    


usage:
   mpirun -np 4 \
    -H server1:1,server2:1,server3:1,server4:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
    cifar10_train.py [cifar10_train.yaml]

TODO: 
 - save checkpiont by params
'''

## TODO move single file
def update_param(pipe):
    train_file = './cifar10_train.yaml' if len(sys.argv) == 1 else sys.argv[1]

    with open(train_file, 'r') as f:
        param = yaml.load(f)
    
    #todo compute minbatch by gpumem and model 

    param['all_step'] = max(int(param.epoch * param.train_nums / param.minibatch), 10)
    param['step_per_epoch'] = max(int(param.train_nums / param.minibatch), 1)
    param['batch_size'] = max(int(param.minibatch / pipe.gpu_nums), 1)
    param['minibatch'] = param.batch_size*pipe.gpu_nums

    #this is nessary compute lr with epoch 
    param['train_nums'] = (param.train_nums // param.minibatch) * param.minibatch
    
    #update checkoint name for easy run mulitimes
    param['checkpoint'] = "checkpoint/" + "_".join([ param.name, "batch"+str(param.minibatch), "layer"+str(param.resnet_layer)])
    
    if not os.path.exists(param['checkpoint']):
        os.system("mkdir -p " + param['checkpoint'])

    tf.logging.info("Param load from " + train_file)
    return param

def train_dataset(param):
    train_set, vaild_set, eval__set = cifar10_dataset(param.data_path, param.batch_size, param.epoch)
    return train_set

def eval_dataset(param):
    train_set, vaild_set, eval__set = cifar10_dataset(param.data_path, param.batch_size, param.epoch)
    return eval__set    
    
def accuracy(labels, predicts, topk):
    return tf.reduce_sum(tf.cast(tf.nn.in_top_k(predicts, labels, topk), tf.float32))

def main(argv):

    pipe  = multipipeline.Pipeline()
    # 1. config load, as hypter params should define in yaml config
    param = update_param(pipe) 

    #print(tf.get_default_graph().get_operations())
    # 2. selet a model, dataset, lr, opt, and so on,  as these can be enumeration.
    create_model_func = official_model.Cifar10Model(param.resnet_layer, param.class_num) 
    
    evaluater = Evaluater(param, eval_dataset, 
                          modelfun = lambda image : create_model_func(image, False),
                          accuracyfun = lambda labels, predicts: accuracy(labels, predicts, 1))



    global_step = tf.train.get_or_create_global_step()
    lr = PiecewiseLR(param)
    opt = tf.train.MomentumOptimizer(lr, param.momentum)    
    
    #3 set up graph these should not modify
    #3.1 set_up dataset
    train_set = train_dataset(param)
    device_dataset = pipe.setup_dataset(train_set)

    #3.2 set_up model and loss    
    device_labels, device_predicts = pipe.setup_model(device_dataset, create_model_func)
    
    
    def compute_loss_func(labels, logits):
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        return tf.reduce_mean(loss)
    device_losses, sum_loss = pipe.setup_loss(device_labels, device_predicts, compute_loss_func, param.weight_decay)
    


    top1 = pipe.setup_reduce(device_labels, device_predicts, lambda x,y:accuracy(x,y, 1), use_mean=True)
    top5 = pipe.setup_reduce(device_labels, device_predicts, lambda x,y:accuracy(x,y, 5), use_mean=True)

    
    #3.3 set_up gradient compute and update
    train_op = pipe.setup_train(device_losses, opt)
    
    
    #TODO add restore hook
    
    hooks = pipe.get_hook() + [
         tf.train.StopAtStepHook(last_step = param.all_step),
         train_hook.SaverHook(param, save_every_n_steps=1000, evaluater=evaluater),
         train_hook.TrainStateHook(param, lr, sum_loss, 
                                    {'batch_top1': top1, 'batch_top5': top5},
                                   every_sec = 10)     
    ]
    
    

    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    
    config.gpu_options.allow_growth = True
    config.use_per_session_threads = True
    config.isolate_session_state = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4

    
    # start train loop 
    scaffold = InitScaffold(param)
    with tf.train.MonitoredTrainingSession(hooks=hooks,scaffold = scaffold,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run([train_op])
            
if __name__ == "__main__":
    tf.app.run()
