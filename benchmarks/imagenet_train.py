#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# keep easy understand and simple to extend.
# same as keras https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py 


import sys
import os
import logging

# modules
sys.path.append('.')

#import before tf
import logger
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import config_pb2

from param         import Configure
from data_set      import imagenet_dataset
from model         import official_model
from learning_rate import PiecewiseLR
from pipeline      import multipipeline
from initialize    import InitScaffold
from evaluater     import Evaluater
import train_hook

#tf.logging.set_verbosity(tf.logging.DEBUG)

'''
setup:
    0. build tensorflow depency
       apt install libibverbs-dev
       apt-get install cuda-command-line-tools
       export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64
       

       apt-get install --reinstall python3-minimal python-lockfile
       apt-get install python3-numpy python3-dev python3-pip python3-wheel
       
       #missing input file '@local_config_nccl//:nccl/NCCL-SLA.txt
       # cp to third_party/nccl/NCCL-SLA.txt
       
       bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
       bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
       pip3 install -U tensorflow-1.8.0-cp35-cp35m-linux_x86_64.whl 

 

       pip3 install -U pyyaml
       pip3 install -U torch
    
       pip3 uninstall tensorflow-gpu
       pip3 uninstall tensorflow
       pip3 install  --no-cache-dir -U tensorflow-gpu
    

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
        # fix build error: : fatal error: tensorflow/compiler/xla/statusor.h: No such file or directory
        /usr/local/lib/python3.5/dist-packages/tensorflow/include/tensorflow# cp -r /tensorflow/tensorflow/compiler  .

        sudo -E HOROVOD_GPU_ALLREDUCE=NCCL  HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir -U horovod


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



def train_dataset(param):
    train_set, _ = imagenet_dataset(param.train_path, param.batch_size)

    return train_set

def eval_dataset(param):
    _, eval__set = imagenet_dataset(param.validation_path, param.batch_size)
    return eval__set    
    
def accuracy(labels, predicts, topk):
    return tf.reduce_sum(tf.cast(tf.nn.in_top_k(predicts, labels, topk), tf.float32))

def main(argv):
    conf_file = './imagenet_train.yaml' if len(sys.argv) == 1 else sys.argv[1]
    
   
    # 1. config load, as hypter params should define in yaml config
    config = Configure(conf_file)
    config.reconfigure()
    param = config.param

    # 2. selet a model, dataset, lr, opt, and so on,  as these can be enumeration.
    create_model_func = official_model.ImageNetModel(param.resnet_layer, param.class_num) 
    
    if param.validation_enable:
        evaluater = Evaluater(param, eval_dataset, 
                          modelfun = lambda image : create_model_func(image, False),
                          accuracyfun = lambda labels, predicts: accuracy(labels, predicts, 1))
    else:
        evaluater = None

    pipe  = multipipeline.Pipeline(param)
    
    with tf.device('/device:CPU:0'), tf.name_scope('cpu_0') as op_scope:
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
    device_losses, train_loss, l2_loss = pipe.setup_loss(device_labels, device_predicts, compute_loss_func, param.weight_decay)
    


    top1 = pipe.setup_reduce(device_labels, device_predicts, lambda x,y:accuracy(x,y, 1), use_mean=True)
    top5 = pipe.setup_reduce(device_labels, device_predicts, lambda x,y:accuracy(x,y, 5), use_mean=True)
    
    tf.summary.scalar('top1', top1)
    tf.summary.scalar('top5', top5)

    
    #3.3 set_up gradient compute and update
    train_op = pipe.setup_train(device_losses, opt)
    
    hooks = pipe.get_hook() + [
        tf.train.StopAtStepHook(last_step = param.all_step),
        train_hook.SaverHook(param, save_every_n_steps=10000, evaluater=evaluater),
        train_hook.TrainStateHook(param, lr, train_loss, l2_loss, 
                                    {'batch_top1': top1, 'batch_top5': top5},
                                   every_sec = 15),
        train_hook.ProfilerHook(save_steps=200, output_dir=param.checkpoint)
        #train_hook.SummaryHook(path=param.checkpoint)
    ]
    
    logging.info("set up hook done") 
 

    
    # refer from tensorflow/core/protobuf/config.proto
    config = tf.ConfigProto()

    config.allow_soft_placement=True
    config.log_device_placement=False
    
    # cpu thread auto set    
    #config.intra_op_parallelism_threads=0
    #config.inter_op_parallelism_threads=0
    #config.session_inter_op_thread_pool.num_threads=0
    #config.session_inter_op_thread_pool.global_name='train'

    #session 
    config.use_per_session_threads = True
    config.isolate_session_state = True
    
    # gpu
    config.gpu_options.allow_growth = True
    config.gpu_options.force_gpu_compatible=False
    # this disable nccl ?
    #config.gpu_options.experimental.use_unified_memory=True
    #config.gpu_options.per_process_gpu_memory_fraction = 1.0

    # graph
    #config.graph_options.enable_recv_scheduling = True # no used
    #config.graph_options.build_cost_model=1
    #config.graph_options.build_cost_model_after=1
    #config.graph_options.infer_shapes=True
    #config.graph_options.enable_bfloat16_sendrecv=False
    config.graph_options.optimizer_options.do_common_subexpression_elimination = True
    config.graph_options.optimizer_options.max_folded_constant_in_bytes = 0 # default 10M
    #config.graph_options.optimizer_options.do_function_inlining = True # default 10M

    #config.graph_options.optimizer_options.opt_level = config_pb2.OptimizerOptions.L1
    #config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.ON_1
   
    # default on
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.ON
    #config.graph_options.rewrite_options.optimizers='autoparallel'
    
    # start train loop 
    scaffold = InitScaffold(param)
    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                            scaffold = scaffold,
                                            config=config) as mon_sess:
        
        #pipe.vgr.debug_cross_device_op()
                                      
        while not mon_sess.should_stop():
            mon_sess.run([train_op])
            
if __name__ == "__main__":
    tf.app.run()
