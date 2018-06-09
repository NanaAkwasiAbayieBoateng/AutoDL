import os
import sys
import yaml
import logging

import tensorflow as tf

from device_manager import gpu



from name_dict import NameDict

class Configure:
    def __init__(self, yamlfile):
        self.conf_file = yamlfile
        with open(self.conf_file, 'r') as f:
           self.param = yaml.load(f)
        
    def reconfigure(self):
        param = self.param

        param.gpu_nums = gpu.get_nr_gpu()
        
        # TODO compute the minibatch by GPU MEM
        param.all_step = max(int(param.epoch * param.train_nums / param.minibatch), 10)
        param.step_per_epoch = max(int(param.train_nums / param.minibatch), 1)
        param.batch_size = max(int(param.minibatch / param.gpu_nums), 1)
        param.minibatch = param.batch_size*param.gpu_nums

        #this is nessary compute lr with epoch 
        param.train_nums = (param.train_nums // param.minibatch) * param.minibatch
    
        #update checkoint name for easy run mulitimes
        param.checkpoint = "checkpoint/" + "_".join([ param.name, "batch"+str(param.minibatch), "layer"+str(param.resnet_layer)])
    
        if not os.path.exists(param['checkpoint']):
             os.system("mkdir -p " + param['checkpoint'])
        
        logging.info("reconfigure:"+repr(self.param))

        