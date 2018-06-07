import os
import time
from datetime import datetime

#import logging
#logging.basicConfig(level = logging.INFO,format = '%(asctime)s:%(levelname)s - %(message)s')

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks


from multiprocessing import Process,Queue,Event

class EvaluateRunner:
    '''
    use last n cpu to eval the test or vaild data
    as we start a new graph, the parent graph should copy on write, also can be done by `graph container api`
    '''
    def __init__(self, handler):
        self.queue = Queue()
        self.event = Event()
        self.handle = hander

        self.process = Process(target=self.run, args=())
        self.process.start()
        self.event.wait()

    def stop(self):
        self.queue.put(-1)
        self.event.wait()

    def async_run(self, step):
        self.queue.put(step)

    def run(self):
        # notify start
        '''
        as this is process the default graph will copy to this process
        '''
        self.handler.setup()
        self.event.set()
  
        
        while True:
           step = self.queue.get()
           if step < 0 :
               break
           self.handler.do(step)
           #do process                      
        # notify stop
        self.event.set()  
  


class EvaluateHook(tf.train.SessionRunHook):
    '''
    trigger in main session
    '''
    def __init__(self, runner,every_n_steps ):
        self.runner = runner
        self.every_n_steps = every_n_steps

    def begin(self):
        self._g_step = tf.train.get_global_step()
 
    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs([self._g_step])
    
    def after_run(self, run_context, run_values):
        step = run_values.results[0]
        if step > 0 and (step % self._save_every_n_steps) == 0:
            self.runner.async_run(step)

    def end(self, session):
        self.runner.stop()


class Evaluater:
    
    def __init__(self, param, dataset, modelfun, accuracyfun):
        # update thread num 
        self.threadsnum = param.eval_threadsnum
        if self.threadsnum <= 1:
            self.threadsnum = max(os.cpu_count() / 4, 2)
            param.eval_threadsnum = self.threadsnum
        
        self.dataset = dataset
        self.modelfun = modelfun
        self.accuracyfun = accuracyfun
        # start a new process
        self.runner = EvaluateRunner(self)
    
    def EvaluateHook(self, every_n_steps=1000):
        return EvaluateHook(self, every_n_steps)
    
    def setup(self):
        '''
        this run in child process
        '''

        self.iterator = self.dataset.make_initializable_iterator()
        image, label = iterator.get_next()

        prediction = self.modelfun(image, False)
        probs = tf.nn.softmax(prediction)
        self.top1= self.accuracyfun(label, probs)
            
        #for ROC refer: https://blog.csdn.net/jiangjieqazwsx/article/details/52262389
        #os.environ['CUDA_VISIBLE_DEVICES']=''
        self.config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        self.config.gpu_options.visible_device_list = []
        # see ./tensorflow/docs_src/performance/performance_guide.md 
        # Nodes that can use multiple threads to
        # parallelize their execution will schedule the individual pieces into this pool.
        self.config.intra_op_parallelism_threads = self.threadsnum
        #All ready nodes are scheduled in this pool.
        self.config.inter_op_parallelism_threads = self.threadsnum
        self.config.device_count['CPU'] = self.threadsnum
        self.config.device_count['GPU'] = 0 

        self.config.use_per_session_threads = True
        self.config.isolate_session_state = True     
    
    def do(self, step):
        self.param.model_init.pre_train_step = step
        scaffold = InitScaffold(self.param)
        
        # start e loop
        with tf.train.MonitoredTrainingSession(scaffold=scaffold, config=config) as mon_sess:
            allnum = 0
            top1 = 0
            while True:
                top1 += mon_sess.run([top1])
                allnum += self.param.minibatch
        
        accury = 100.0*top1 / allnum
        tf.logging.info(all top1 {accury:.2f}".format(accury=accury))