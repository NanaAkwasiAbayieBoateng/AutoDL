import tensorflow as tf
import operator 

from tensorflow.python.framework import device as pydev


def deviceTosepc(deviceid, device_type='GPU'):
    device_spec = pydev.DeviceSpec(job='localhost', 
                                replica=0,
                                task=0,
                                device_type=device_type,
                                device_index=deviceid)
    return device_spec.to_string()


class ReplicatePlacement:
    '''
    placement variable to  each_device  
    '''
    def __init__(self, gpu_nums):
        self.gpu_nums = gpu_nums
        self.worker_devices = [deviceTosepc(i) for i in range(self.gpu_nums)]
        self.device_type = 'GPU'
        
    def get_create_scope(self, deviceid):
        
        # because will placement from 0 to N
        var_scope = tf.variable_scope('gpu_%s' % deviceid, reuse=False)
        op_scope = tf.name_scope('tower_%s' % deviceid)
        device_scope = tf.device(self.worker_devices[deviceid])
        return var_scope, op_scope, device_scope
    
    def get_trainable_variable(self, i=None):
        if i == None:
            return tf.trainable_variables()
        device_spec = self.worker_devices[i]
        
        return [
            v for v in tf.trainable_variables() if v.device == device_spec
        ]

    def get_brocastop(self):
        # Copy initialized values for variables on GPU 0 to other GPUs.
        global_vars = tf.global_variables()
        var_by_name = dict([(v.name, v) for v in global_vars])
        post_init_ops = []
        for v in global_vars:
            split_name = v.name.split('/')
            # TODO(b/62630508): use more specific prefix than v or v0.
            if split_name[0] == 'gpu_0' or not v.name.startswith('gpu_'):
                continue
            split_name[0] = 'gpu_0'
            copy_from = var_by_name['/'.join(split_name)]
            post_init_ops.append(v.assign(copy_from.read_value()))
        return post_init_ops


class BalancePlacement:
    
    def __init__(self, gpu_nums):
        self.gpu_nums = gpu_nums
        self.worker_devices = [deviceTosepc(i) for i in range(self.gpu_nums)]
        self.sizes = [0] * self.gpu_nums
    
    def _device_getter(self, currend_deviceid):
        #op placement to gpu
        self.varable_ops = ['Variable', 'VariableV2', 'VarHandleOp']

        def _local_device_chooser(op):
            
            default_spec = pydev.DeviceSpec(job='localhost', 
                                replica=0,
                                task=0,
                                device_type='GPU',
                                device_index=currend_deviceid)

            strategy = op.name + "," + op.type + "," + op.device +"->"
            # not overwrite
            if op.device:
                op_spec = pydev.DeviceSpec.from_string(op.device)
                default_spec.merge_from(op_spec)
                strategy = strategy + ", by op.device" + op.device
            # if operation placement or const to worker_device
            elif op.type not in self.varable_ops:
                strategy = strategy + ", by op.type" + str(currend_deviceid)
            
            # as we only update the first tower BN, so all bn variable at deviceid0
            elif 'BatchNorm' in op.name:
                default_spec.device_index = 0
                strategy = strategy + ", by BatchNorm" + str(currend_deviceid)
            else:
                # min_shapesize
                device_index, _ = min(
                     enumerate(self.sizes), key=operator.itemgetter(1))
                default_spec.device_index = device_index

                # as varable op, which should only has one output
                var_size = op.outputs[0].get_shape().num_elements()
                self.sizes[device_index] += var_size

                strategy = strategy + ", by shape%s:" % op.outputs[0].get_shape() + str(device_index)
                #print(strategy)
            return default_spec.to_string()

        return _local_device_chooser
    
    def get_create_scope(self, deviceid):
        # because will placement from 0 to N
        var_scope = tf.variable_scope('gpu', reuse=deviceid != 0)
        op_scope = tf.name_scope('tower_%s' % deviceid)
        device_scope = tf.device(self._device_getter(deviceid))
        return var_scope, op_scope, device_scope
    
    def get_trainable_variable(self, i):
        return [ v for v in tf.trainable_variables()]

    def get_brocastop(self):
        return []
