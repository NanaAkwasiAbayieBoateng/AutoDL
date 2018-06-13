import tensorflow as tf
import operator
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import device as pydev


def deviceTosepc(deviceid, device_type='GPU'):
    device_spec = pydev.DeviceSpec(job='localhost',
                                replica=0,
                                task=0,
                                device_type=device_type,
                                device_index=deviceid)
    return device_spec.to_string()


class MultGPUPlacement:
    def __init__(self, gpu_nums):
        self.gpu_nums = gpu_nums
        self.worker_devices = [deviceTosepc(i) for i in range(self.gpu_nums)]
        self.device_type = 'GPU'
        self.untrainable_vars = {}
    
    def _variable_getter(self, getter, name, *args, **kwargs):
        """ find out the BN mean and variance, placement the GPU0 """
        if 'trainable' in kwargs and not kwargs['trainable']:
            var = self.untrainable_vars.get(name, None)
            if var is None:
                var = getter(name, *args, **kwargs)
                self.untrainable_vars[name] = var
                #logging.info("untrainable:" + str(var))
            return var
        else:
            return getter(name, *args, **kwargs)
        
    def get_gradients_varmap(self):
        raise NotImplementedError("Must reimplement scope")
        """return the deviceid need to compute, return map to cache multi get """
     def get_device_varmap(self, deviceid):
        raise NotImplementedError("Must reimplement scope")
    
    def get_update_ops(self):
        """For BN  each tower update """
        ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        return ops

    def get_create_scope(self, deviceid):
        raise NotImplementedError("Must reimplement scope")
    
    def get_brocastop(self):
        raise NotImplementedError("Must reimplement scope")

    def _device_getter(self, currend_deviceid):
        raise NotImplementedError("Must reimplement scope")
    
    
    def debug_cross_device_op(self):
        def filter(op):
            if 'save' in op.name or 'report_uninitialized' in op.name or 'Initializer' in op.name:
                return False
            return True
        
        ops =  (o for o in tf.get_default_graph().get_operations() if filter(o))
        
        unplacentment = []
        unkown = []
        cpufromgpu = []
        gpufromcpu = []
        gpufromgpu = []

        for op in ops:
            if len(op.device) < 1:
                unplacentment.append(op)
                continue
            if 'CPU' in op.device:                
                for i in (o for o in op.inputs if filter(o)):
                    if 'GPU' in i.device:
                        cpufromgpu.append((i, op))
                continue

            if 'GPU' in op.device:
                for i in (o for o in op.inputs if filter(o)):
                    if 'CPU' in i.device:
                        gpufromcpu.append((i, op))
                    elif ('GPU' in i.device) and (op.device != i.device):
                        gpufromgpu.append((i, op))
                continue

            unkown.append(op)
        
        with  open('debug.log', 'w') as f:
            strs = "unkown\n" + '\n--'.join([o.name +":" +o.device for o in unkown])
            f.write(strs)
            strs = "\nunplacentment\n" + '\n++'.join([o.name +":" +o.device for o in unplacentment])
            f.write(strs)
            strs = "\ncpufromgpu\n" + '\n--'.join([i.name +":" +i.device +"->"+o.name +":"+o.device  for i,o in cpufromgpu])
            f.write(strs)
            strs = "\ngpufromcpu\n" + '\n++'.join([i.name +":" +i.device +"->"+o.name +":"+o.device  for i,o in gpufromcpu])
            f.write(strs)
            strs = "\ngpufromgpu\n" + '\n--'.join([i.name +":" +i.device +"->"+o.name +":"+o.device  for i,o in gpufromgpu])
            f.write(strs)
              
        
        

'''
ReplicatePlacement:
FOR  NVLINK GPU every GPU is P2pP 300GB, read network and gpu update very fast
Place var to unique GPU, other gpu has cache
    1. when compute grad is done unique GPU allreduce.
    2. when forward, var is read from unique GPU

varnum: M, GPUnum: N
allreduce single max: (N-1)*M  # only reduce from other
allreduce lateny = 1read network + 1gpu update

master-slave: master: 2(N-1)*M, slave: 2M # master reduce then brocast
allreduce lateny = 1read network +  1gpu update + 1read network 
'''

'''
BalancePlacement:
FOR  PCI-E GPU every GPU bandwith is limmit, 
Place every GPU a copy so only grad is commiucate.

varnum: M, GPUnum: N
allreduce + readcopy single max: (N-1)*M/N + (M - M/N) = 2M* (N-1)/N  # only reduce from other
lateny = (1read network +  1gpu update + 1read network)/N

'''

class ReplicatePlacement(MultGPUPlacement):

    def __init__(self, gpu_nums):
        super().__init__(gpu_nums)

    def get_create_scope(self, deviceid):
        # because will placement from 0 to N
        # var scope control the graph vars, as each gpu has a single var or shared a var
        # device_scope placement the var to which gpu, affect the var read update copy
        
        # for replicate every gpu has own var and operation
        var_scope = tf.variable_scope('gpu_%d' % deviceid, custom_getter=self._variable_getter, reuse=False)
        op_scope = tf.name_scope('tower_%s' % deviceid)
        device_scope = tf.device(self.worker_devices[deviceid])
        return var_scope, op_scope, device_scope
    

    def get_gradients_varmap(self):        
        """return the deviceid need to compute gradients, for replciate each tower only compute its var """
        gradients_varmap = {}
        for v in tf.trainable_variables():
            spec = pydev.DeviceSpec.from_string(v.device)
            gradients_varmap.setdefault(spec.device_index, []).append(v)
        return gradients_varmap            

    def get_device_varmap(self, deviceid):      
        """used to compute the normal, only gradients need to compute l2 loss"""
        
        gradients_varmap = {}
        for v in tf.trainable_variables():
            spec = pydev.DeviceSpec.from_string(v.device)
            gradients_varmap.setdefault(spec.device_index, []).append(v)
        return gradients_varmap   
    

    def get_update_ops(self):
        """For BN  each tower update """
        ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        ### TODO brocast the BN
        # 1 update local
        # 2 brocast from master
        return ops
    
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
            
            # used gpu_0 as master
            split_name[0] = 'gpu_0'
            copy_from = var_by_name['/'.join(split_name)]
            post_init_ops.append(v.assign(copy_from.read_value()))
        return post_init_ops


class BalancePlacement(MultGPUPlacement):

    def __init__(self, gpu_nums):
        super().__init__(gpu_nums)
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

            #node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
            strategy = op.name + "," + op.type + "," + op.device +"->"
            # not overwrite
            if op.device:
                op_spec = pydev.DeviceSpec.from_string(op.device)
                default_spec.merge_from(op_spec)
                strategy = strategy + ", by op.device:" + op.device
            # if operation placement or const to worker_device
            elif op.type not in self.varable_ops:
                strategy = strategy + ", by op.type:" + str(currend_deviceid)

            # as we only update the first tower BN, so all bn variable at deviceid0
            #elif node_def.name in self.untrainable_vars:
            #    default_spec.device_index = 0
            #    strategy = strategy + ", by untrainable_vars:" + str(currend_deviceid)
            #    assert op.outputs[0].name == node_def.name
            #    assert op.outputs.size() == 1
            else:
                # min_shapesize
                device_index, _ = min(
                     enumerate(self.sizes), key=operator.itemgetter(1))
                default_spec.device_index = device_index

                # as varable op, which should only has one output
                assert len(op.outputs)== 1
                
                var_size = op.outputs[0].get_shape().num_elements()
                self.sizes[device_index] += var_size

                strategy = strategy + ", by shape %s:" % op.outputs[0].get_shape().as_list() + str(device_index)
                
            return default_spec.to_string()

        return _local_device_chooser

    def get_create_scope(self, deviceid):
        # because will placement from 0 to N
        var_scope = tf.variable_scope('gpu', custom_getter=self._variable_getter, reuse=(deviceid!=0))
        op_scope = tf.name_scope('tower_%s' % deviceid)
        device_scope = tf.device(self._device_getter(deviceid))
        return var_scope, op_scope, device_scope

    def get_gradients_varmap(self):        
        """return the deviceid need to compute gradients, for replciate each tower only compute its var """
        vars = tf.trainable_variables()
        gradients_varmap = {i:var for i in range(self.gpu_nums):}
        return gradients_varmap            

    def get_device_varmap(self, deviceid):      
        """used to compute the normal,  every gradient variable  need to compute l2 loss"""        
        vars = tf.trainable_variables()
        gradients_varmap = {i:var for i in range(self.gpu_nums):}
        return gradients_varmap  

    def get_brocastop(self):
        return []
