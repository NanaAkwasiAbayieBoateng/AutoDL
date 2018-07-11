
import os
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.framework import device as pydev

#single node reducer
from tensorflow.contrib import nccl

##  multinode  reducer
import horovod.tensorflow as hvd

from device_manager import gpu
from pipeline import var_placement

class PipelineHook(session_run_hook.SessionRunHook):
    '''
    prefer just the hide logic to use hook, add some debug msg
    '''
    def __init__(self, node_coord=None):
        self.init_ops = []
        self.warm_ops = []
        self.run_ops = []
        self.fetches = {}
        self.root_rank = 0
        self.rank = 0

        self.node_coord = node_coord
        if node_coord:
            self.rank  = node_coord.rank()

        ## root rank is zero

    def add_logtensor(self, key, tensor):
        self.fetches[key]=tensor

    def begin(self):
        '''
         The hook can modify the graph by adding new operations to it,
         Second call of begin() on the same graph, should not change the graph.
        '''
        #self.node_coord.init()

        ## device ? cpu
        if self.node_coord:
            self.init_ops += [self.node_coord.broadcast_global_variables(self.root_rank)]

        self.global_step = tf.train.get_global_step()
        self.run_ops += [self.global_step.assign_add(1)]



    def end(self, session):
        '''when session.run() raises OutOfRangeError or StopIteration.
         In that case end() is called but after_run() is not called.
        '''
        return super().end(session)

    def after_create_session(self, session, coord):
        '''the graph is finalized and ops can no longer be added to the graph.
        '''
        logging.info("run init_ops")
        session.run(self.init_ops)
        logging.info("run warn_ops")
        session.run(self.warm_ops)
        self.fetches.update({'global_step':self.global_step, 'run_ops':self.run_ops})
       
        return super().after_create_session(session, coord)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        # extend session.run(ops) so the ops can be excute parallel
       
        if self.rank != self.root_rank:
            return basic_session_run_hooks.SessionRunArgs(fetches={})
            
        # only root print log
        return basic_session_run_hooks.SessionRunArgs(fetches=self.fetches)

    def after_run(self, run_context, run_values):
        
        if self.rank != self.root_rank:
            return
        
        # only root print log
        _ = run_values.results['global_step']
        for k in self.fetches.keys():
            if k == 'run_ops' or k == 'global_step' :
                continue

            print('%s:%s' %(k, run_values.results[k]))
'''
the GPU topo affect reduce algthrithm

topo get:
   nvidia-smi nvlink -s -p -i 0
name get:
   lspci -nn |grep NVIDIA

'''

## TODO: wrapper a reducer
## as nccl now not support reduce_sum op, in graph,
## reduce only can use copy
def single_reduce(tensors_across_devices, use_mean=True):
    """Does an all-reduce of a list of tensors by copying to the current device.
    The tensors are copied to the current device and then reduced.
    Args:
      tensors_across_devices: A list of tensors, each on a different device.
      use_mean: Whether to take the mean of the tensors instead of a sum:
    Returns:
      A reduced tensor on the current device.
    """
    if type(tensors_across_devices) != list:
         tensors_across_devices = list(tensors_across_devices)

    if len(tensors_across_devices) == 1:
        return tensors_across_devices[0]
    #as not each GPU has P2P over NVLINK, so this willbe be bottleneck
    #TODO use tensorflow/contrib/all_reduce/python/all_reduce.py

    reduced_tensor = tf.add_n(tensors_across_devices)
    #reduced_tensor = nccl.reduce_sum(tensors_across_devices)

    if use_mean:
        reduced_tensor *= 1.0 / len(tensors_across_devices)
    return reduced_tensor



def nccl_all_reduce(tensors_across_devices, use_mean=True):
    """Does an all-reduce of a list of tensors by copying to the current device.
    The tensors are copied to the current device and then reduced.
    Args:
      tensors_across_devices: A list of tensors, each on a different device.
      use_mean: Whether to take the mean of the tensors instead of a sum:
    Returns:
      A reduced tensor on the current device.
    """
    if type(tensors_across_devices) != list:
         tensors_across_devices = list(tensors_across_devices)

    if len(tensors_across_devices) == 1:
        return tensors_across_devices[0]

    reduced_tensor = nccl.all_sum(tensors_across_devices)
    if not use_mean:
        return reduced_tensor
    
    assert len(reduced_tensor) == len(tensors_across_devices)

    results = []
    for t, v in zip(reduced_tensor, tensors_across_devices):

        assert t.device == v.device, "t:%s, v:%s should same device" %(t, v)
        with ops.colocate_with(v):
            reduced_tensor = t * 1.0 / len(tensors_across_devices)
            results.append(reduced_tensor)
        
    return results




class Pipeline:
    def __init__(self, param):
        self.workers = param.workers
        self.gpu_nums = gpu.get_nr_gpu()
        
        ##  placement stragety, allreduce + nccl + rmda
        if param.placement == 'replicate':
            self.vgr = var_placement.ReplicatePlacement(self.gpu_nums)
        else:
            self.vgr = var_placement.BalancePlacement(self.gpu_nums)
            param.placement = 'balance'

        # use Horovod as multi-node reducer
        self.root_rank = 0
        if len(self.workers) > 1:
            hvd.init()
            self.node_reduce = hvd.allreduce
            self.node_coord = hvd
            self.rank = hvd.rank()   
        else:
            self.node_reduce = None
            self.node_coord = None
            self.rank = 0
            
        
        self.single_reduce = single_reduce
        #TODO use xring allreduce ?
        self.all_reduce = nccl_all_reduce

        
        self.hook = PipelineHook(self.node_coord)
  
        self.gpu_devices = self.vgr.worker_devices
        self.cpu_devices = ['/device:CPU:0']
      

        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        # as cuda has single dtoH ,htod process, no use for too much threads
        os.environ['TF_GPU_THREAD_COUNT'] = '2' 
        # see https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html
        
        #start docker --shm-size=1g --ulimit memlock=-1
        os.environ['NCCL_DEBUG']='INFO'
        os.environ['NCCL_BUFFIZE']='67108864' # 64M
        os.environ['NCCL_NTHREADS']='256'     # only be 64, 128 or 256. Ignoring     
        #os.environ['NCCL_MAX_NRINGS']='4'     # 4 nvlink for each gpu
        os.environ['NCCL_SINGLE_RING_THRESHOLD']='4096' #4K 

        # Default to two threads. One for device compute and another for memory copies.
        logging.info("pipeline init done, rank:%d gpu:%s placement:%s" % (self.rank, self.vgr.worker_devices, param.placement))


    def get_hook(self):
        return [self.hook]

    def setup_dataset(self, dataset):
        """
        TODO benchmark stage, Impelement the DFS Reader
        """
        ds_iter = dataset.make_initializable_iterator()
        # use stage overhide the data from cpu to gpu copy

        enqueue_ops = []
        device_dataset = {}
        for i, dev in enumerate(self.gpu_devices):
            assert "GPU:%s"%i in dev, "devices must from 0 to N"
            # should has diff input
            image_batch, label_batch  = ds_iter.get_next()
           
            dtypes=[image_batch.dtype, label_batch.dtype]
            shapes=[image_batch.get_shape(), label_batch.get_shape()]
            
            # cpu stage
            with tf.device(self.cpu_devices[0]), tf.name_scope('cpustage_%d' %i) as op_scope:

                gpu_copy_stage = tf.contrib.staging.StagingArea(dtypes=dtypes, shapes=shapes)
                gpu_copy_stage_op = gpu_copy_stage.put([image_batch, label_batch])
                copy_stage_outputs = gpu_copy_stage.get()
            
            # gpu stage
            with tf.device(dev), tf.name_scope('gpustage_%d' %i) as op_scope:
                gpu_compute_stage = tf.contrib.staging.StagingArea(dtypes=dtypes, shapes=shapes)
                gpu_compute_stage_op = gpu_compute_stage.put(copy_stage_outputs)
                compute_stage_outputs = gpu_compute_stage.get()
               
            enqueue_ops += [gpu_copy_stage_op, gpu_compute_stage_op]
            device_dataset[i] = compute_stage_outputs

        # add op to hook
        self.hook.init_ops += [ds_iter.initializer]
        self.hook.warm_ops += enqueue_ops

        self.hook.run_ops += enqueue_ops
        
        logging.info("setup dataset done")
        return device_dataset

    def setup_model(self, device_dataset, create_model_func, isTrain=True):
        device_label = {}
        device_predict = {}

        for i, dev in enumerate(self.gpu_devices):
            var_scope, op_scope, device_scope = self.vgr.get_create_scope(i)
            with var_scope, op_scope, device_scope:
                image, label = device_dataset[i]
                assert label.device == dev
                assert image.device == dev
                
                #tf.summary.image("image_%s"%i, image)
                predict = create_model_func(image, isTrain)            
            device_label[i] = label
            device_predict[i] = predict

        # add to updateop
        self.hook.init_ops += self.vgr.get_brocastop()
        
        # update each tower
        self.hook.run_ops += self.vgr.get_update_ops()

        logging.info("setup model done")
        return device_label, device_predict

    def setup_loss(self, device_labels, device_predicts, compute_func, weight_decay=0.0002):
        '''
        default we add l2 loss
        '''
        device_losses = {}
        train_losses = []
        l2_losses = []
        
        tower_trainvars = self.vgr.get_device_varmap()
        for i, predict in device_predicts.items():

            labels = device_labels[i]
            assert 'GPU:%s'%i in predict.device
            assert labels.device == predict.device

            with tf.device(predict.device), tf.name_scope('loss_stage_%d' %i) as op_scope:
                
                loss  = compute_func(labels=labels, logits=predict)
                
                # for eliminate commicate, each gpu compute itself  and l2 loss should be same
                norm_vars = tower_trainvars[i]
                l2loss = [tf.nn.l2_loss(v) for v in norm_vars]
                l2loss = weight_decay * tf.add_n(l2loss)
                
                train_losses.append(loss)
                l2_losses.append(l2loss)

                #self.hook.add_logtensor("train_loss_%s"%i, loss)       
                #self.hook.add_logtensor("l2_loss_%s"%i, l2loss)       
                
                # each gpu compuie his loss
                device_losses[i] = loss + l2loss        

        #reduce gpus to cpu 
        with tf.device(self.cpu_devices[0]):
            sum_loss   = self.single_reduce(device_losses.values(), use_mean=True)
            train_loss = self.single_reduce(train_losses, use_mean=True)
            l2_loss    = self.single_reduce(l2_losses, use_mean=True)
        
        # reduce between node
        if self.node_reduce and self.rank == 0:
            sum_loss   = self.node_reduce(sum_loss, average=True, device_dense=self.cpu_devices[0])
            train_loss = self.node_reduce(train_losses, average=True, device_dense=self.cpu_devices[0])
            l2_loss    = self.node_reduce(l2_losses, average=True, device_dense=self.cpu_devices[0])

        logging.info("setup loss done")
        return device_losses, train_loss, l2_loss

    def setup_reduce(self, device_labels, device_predicts, compute_func, use_mean=True):
        '''use_mean
        default we add l2 loss
        '''
        device_values = {}
        for i, predict in device_predicts.items():
            labels = device_labels[i]

            assert 'GPU:%s'%i in predict.device
            assert labels.device == predict.device

            with tf.device(self.gpu_devices[i]), tf.name_scope('reduce_stage_%d' %i) as op_scope:
               device_values[i] = compute_func(labels, predict)

        with tf.device(self.cpu_devices[0]):
            value = self.single_reduce(device_values.values(), use_mean)
        
        if self.node_reduce and self.rank == 0:
            value = self.node_reduce(value, average=use_mean, device_dense=self.cpu_devices[0])
        
        return value

    def setup_train(self, device_losses, opt):
        
        if self.node_coord:
            opt = self.node_coord.DistributedOptimizer(opt)

        grad_map = {}
        var_map  = {}
        #compute and group by var name, we ingore the first tower name
        gradients_varmap = self.vgr.get_gradients_varmap()

        for i, loss in device_losses.items():
            
            with tf.device(self.gpu_devices[i]), tf.name_scope('compute_gradients_stage_%d' %i) as op_scope:

                assert 'GPU:%s'%i in loss.device                
                tower_trainvars = gradients_varmap[i]                
                tower_grad = opt.compute_gradients(loss=loss, var_list=tower_trainvars)
            
            for grad, v in tower_grad:
                # the v.name is gpu*/var_name
                assert v.name.startswith('gpu')
                assert 'gradients' in grad.name
                assert grad.device == v.device

                v_mode_name = '/'.join(v.name.split('/')[1:])

                if grad is not None:
                    # for replicated , each tower has a v
                    # for balance only a var assocate with a gpu
                    var_map.setdefault(v_mode_name, []).append(v)
                    grad_map.setdefault(v_mode_name, []).append(grad)
                else:
                    logging.warn("No found grad for var:"+repr(v))
        
        assert len(grad_map) == len(var_map), "the vars num should be same"
        
        # for all gpus, group grads to device
        device_grad = {}
        for vname, raw_grads in grad_map.items():
            # var_name, var for each gpu, grad for each var            
            vars  = sorted(var_map[vname], key=lambda v : v.device)
            grads = sorted(raw_grads, key=lambda v : v.device)

            grad_device = set([g.device for g in grads])
            assert  len(grad_device) == self.gpu_nums, "each gpu should has a grad, grads:%s" % grads
            
            if len(vars) == 1:
                v = vars[0]
                #var only placement at one gpu 
                with tf.device(v.device):
                    avg_grad = self.single_reduce(grads)
                deviceid = pydev.DeviceSpec.from_string(v.device).device_index
                device_grad.setdefault(deviceid, []).append((avg_grad, v))

            else:
                assert len(vars) == self.gpu_nums, "for replicate each gpu should has a var"

                all_reduce_tensors = self.all_reduce(grads)
                assert len(vars) == len(all_reduce_tensors), "reduce should has same nums"

                for avg_grad, v in zip(all_reduce_tensors, vars):
                    assert avg_grad.device == v.device, "should be same device grad:%s var:%s" % (avg_grad, v)
                    deviceid = pydev.DeviceSpec.from_string(v.device).device_index
                    device_grad.setdefault(deviceid, []).append((avg_grad, v))                 
                
        tran_op = []
        assert len(device_grad.items()) == self.gpu_nums, "each gpu should update gradients"

        for i, gradvars in device_grad.items():
           # check device  
           for grad, v  in gradvars:
               assert 'GPU:%s'%i in v.device
               assert grad.device == v.device 

           with tf.device(v.device),  tf.name_scope('apply_gradients_stage_%d'%i) as op_scope:
               tran_op.append(opt.apply_gradients(gradvars))

        logging.info("setup train done")
        return tran_op
