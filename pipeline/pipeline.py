import os
import tensorflow as tf

from tensorflow.python.framework import tensor_util
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.framework import device as pydev

from tensorflow.contrib import nccl


from device_manager import gpu
from pipeline import var_placement



class PipelineHook(session_run_hook.SessionRunHook):
    '''
    prefer just the hide logic to use hook, add some debug msg
    '''
    def __init__(self):
        self.init_ops = []
        self.run_ops = []
        self.fetches = {}
    
    def add_logtensor(self, key, tensor):
        self.fetches[key]=tensor
        
    def begin(self):
        '''
         The hook can modify the graph by adding new operations to it, 
         Second call of begin() on the same graph, should not change the graph.
        '''
        self.global_step = tf.train.get_or_create_global_step()
        self.run_ops += [self.global_step.assign_add(1)] 

    def end(self, session):
        '''when session.run() raises OutOfRangeError or StopIteration.
         In that case end() is called but after_run() is not called.
        '''
        return super().end(session)
        
    def after_create_session(self, session, coord):
        '''the graph is finalized and ops can no longer be added to the graph.
        '''
        session.run(self.init_ops)
        return super().after_create_session(session, coord)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        # extend session.run(ops) so the ops can be excute parallel
        self.fetches.update({'global_step':self.global_step, 'run_ops':self.run_ops})
        return basic_session_run_hooks.SessionRunArgs(fetches=self.fetches)

    def after_run(self, run_context, run_values):
        _ = run_values.results['global_step']
        for k in self.fetches.keys():
           if k == 'run_ops' or k == 'global_step' :
               continue
               
           print('%s:%s' %(k, run_values.results[k]))


## TODO: wrapper a reducer
def reduce_by_copy(tensors_across_devices, use_mean=True):
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
    reduced_tensor = tf.add_n(tensors_across_devices)
    if use_mean:
        reduced_tensor *= 1.0 / len(tensors_across_devices)
    return reduced_tensor


def reduce_by_nccl(tensors_across_devices, use_mean=True):
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
         
    reduced_tensor = nccl.reduce_sum(tensors_across_devices)
    if use_mean:
        reduced_tensor *= 1.0 / len(tensors_across_devices)
    return reduced_tensor



class Pipeline:
    def __init__(self):
        self.gpu_nums = gpu.get_nr_gpu()
       
        self.vgr = var_placement.ReplicatePlacement(self.gpu_nums)
        self.reduce = reduce_by_copy
       
        self.gpu_devices = self.vgr.worker_devices
        self.cpu_devices = ['/device:CPU:0']
       
        self.hook = PipelineHook()

        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '2'

    
    def get_hook(self):
        return [self.hook]
    
    def setup_dataset(self, dataset):
        
        ds_iter = dataset.make_initializable_iterator()
        image_batch, label_batch  = ds_iter.get_next()
        # use stage overhide the data from cpu to gpu copy
        enqueue_ops = []
        device_dataset = {}
        for i, dev in enumerate(self.gpu_devices):
           with tf.device(dev), tf.name_scope('gpustage_%d' %i) as op_scope:
               gpu_stage = tf.contrib.staging.StagingArea(
                                dtypes=[image_batch.dtype, label_batch.dtype],
                                shapes=[image_batch.get_shape(),label_batch.get_shape()])
                
               put_gpu_op = gpu_stage.put([image_batch, label_batch])
               enqueue_ops.append(put_gpu_op)
               device_dataset[i] = gpu_stage.get()
        # add op to hook 
        self.hook.init_ops.append(ds_iter.initializer)
        self.hook.run_ops += enqueue_ops
        return device_dataset
        
    def setup_model(self, device_dataset, create_model_func, isTrain=True):
        device_label = {}
        device_predict = {}
        for i, dev in enumerate(self.gpu_devices):
            var_scope, op_scope, device_scope = self.vgr.get_create_scope(i) 
            with var_scope, op_scope, device_scope:
                 image, label = device_dataset[i]
                 predict = create_model_func(image, isTrain)
            if i == 0:
                 # as the first construct graph, all bn var is allocated.
                 # every tower may final has some batch, so only update the first tower  
                 update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            device_label[i] = label
            device_predict[i] = predict
            
        # add to updateop 
        self.hook.init_ops += self.vgr.get_brocastop()
        self.hook.run_ops += update_ops
        return device_label, device_predict
    
    def setup_loss(self, device_labels, device_predicts, compute_func, weight_decay=0.0002):
        '''
        default we add l2 loss
        '''
        device_losses = {}
        for i, predict in device_predicts.items():
            
            labels = device_labels[i]
            assert 'gpu_%s'%i in predict.name
            assert labels.device == predict.device
            
            # TODO: for large dataset remove BN var from l2 loss
            tower_trainvars = self.vgr.get_trainable_variable(i)
            
            with tf.device(predict.device):
                loss  = compute_func(labels, predict)
                l2loss = [tf.nn.l2_loss(v) for v in tower_trainvars]
                l2loss = weight_decay * tf.add_n(l2loss)
                total_loss = loss + l2loss
                self.hook.add_logtensor('loss_0', loss)
                self.hook.add_logtensor('l2loss', l2loss)
                self.hook.add_logtensor('total_loss', total_loss)
            
            device_losses[i] = total_loss
        
        with tf.device(self.cpu_devices[0]):
            sum_loss = self.reduce(device_losses.values(), use_mean=False)
        
        self.hook.add_logtensor('sum_loss', sum_loss)
        return device_losses, sum_loss
    
    def setup_reducesum(self, device_labels, device_predicts, compute_func):
        '''
        default we add l2 loss
        '''
        device_values = {}
        for i, predict in device_predicts.items():
            labels = device_labels[i]
            
            assert 'gpu_%s'%i in predict.name
            assert labels.device == predict.device
            
            with tf.device(self.gpu_devices[i]):
               device_values[i] = compute_func(labels, predict)

        with tf.device(self.cpu_devices[0]):
            value = self.reduce(device_values.values(), use_mean=False)
        return value
        
    def setup_train(self, device_losses, opt):
        
        grad_map = {}
        var_map = {}
        #compute and group by var name, we ingore the first tower name
        for i, loss in device_losses.items():
            with tf.device(self.gpu_devices[i]):
                tower_trainvars = self.vgr.get_trainable_variable(i)
                tower_grad = opt.compute_gradients(loss=loss, var_list=tower_trainvars)
            
            for grad, v in tower_grad:
                # for replicated ingore tower name 
                v_mode_name = '/'.join(v.name.split('/')[1:])
                if grad is not None:
                    # for replicated , each tower has a v
                    var_map.setdefault(v_mode_name, []).append(v)
                    grad_map.setdefault(v_mode_name, []).append(grad)
        
        # group to device
        device_grad = {}
        for vname, grads in grad_map.items():
            with tf.device(v.device):
                avg_grad = self.reduce(grads) if len(grads) > 1 else grads[0]
                
            for v in var_map[vname]:
                deviceid = pydev.DeviceSpec.from_string(v.device).device_index
                device_grad.setdefault(deviceid, []).append((avg_grad, v))
        
        tran_op = []
        for i, gradvars in device_grad.items():
            v0 = gradvars[0][1]
            with tf.device(v0.device):
                tran_op.append(opt.apply_gradients(gradvars))

        return tran_op
