'''
copy from tensorpack
'''

import os
from device_manager import nvml

import logging

def get_gpu_devices():
    
    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env is not None:
        return ['/GPU:%s' % i for i in env.split(',')]
    

    # try nvml
    try:
        # Use NVML to query device properties
        with nvml.NVMLContext() as ctx:
            return ['/GPU:%s' % i for i in range(ctx.num_devices())]
    except Exception as e:
        logging.warn("No libnvidia-ml.so.1 found in syslibpath %s" % e)
    
    #this triger a load, may take longtime
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_nr_gpu():
    return len(get_gpu_devices())


if __name__ == '__main__':


    print(get_gpu_devices())