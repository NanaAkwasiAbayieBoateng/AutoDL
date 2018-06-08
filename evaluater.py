import os
import time
from datetime import datetime

#import logging
#logging.basicConfig(level = logging.INFO,format = '%(asctime)s:%(levelname)s - %(message)s')

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks

import multiprocessing
from multiprocessing import Process,Queue,Event

from initialize    import InitScaffold

class EvaluateRunner:
    '''
    use last n cpu to eval the test or vaild data
    as we start a new graph, the parent graph should copy on write, also can be done by `graph container api`
    '''
    def __init__(self, handler, param):
        self.queue = Queue()
        self.event = Event()
        self.handler = handler
        self.param = param
        
        self.setup_status = 0
        self.process = Process(target=self.run, args=())
        self.process.daemon = True
        self.process.start()
        
        "we cannot block main thread"
        #self.event. wait()
        #while self.setup_status == 0:
        #    time.sleep(0)

    def stop(self):
        self.queue.put(-1)
        #self.event.wait()
        self.process.join()


    def run(self):
        # notify start
        '''
        as this is process the default graph will copy to this process
        '''
        self.handler.setup()
        self.setup_status = 1
        self.event.set()
         
        while True:
           step = self.queue.get()
           if step < 0 :
               break
           self.handler.do(step)
           #do process                      
        # notify stop
        self.event.set()  
  

class reporter:
    def __init__(self, path, step, samples, every_sec=10):
        self.step = step
        self.samples = samples
        self.top1 = 0
        self.batches = 0

        self.every_sec = every_sec
        self.start_time = time.time()
        self.last_time = time.time()
        tf.logging.info("evaluate step:{step} path:{path} start".format(step=self.step, path=path))

    def update(self, top1, batches):
        self.top1 += top1
        self.batches += batches

        self.accury   = 100.0 * self.top1  / self.batches
        self.progress = 100.0 * self.batches / self.samples

        if time.time() - self.last_time  >= self.every_sec:
            self.log()
            self.last_time = time.time()
    
    def log(self):
        tf.logging.info("evaluate step:{step} progress:{progress:.2f}% top1:{accury:.2f}"
                            .format(step=self.step, progress=self.progress, accury=self.accury))
      
        
    def end(self):        
        tf.logging.info("evaluate step:{step} finished top1:{accury:.2f} used:{used:.2f}sec"
                  .format(step=self.step, accury=self.accury, used=time.time()-self.start_time))

class Evaluater:
    
    def __init__(self, param, datasetfun, modelfun, accuracyfun):
        # update thread num 
        self.param = param
        self.thread_num = param.eval_thread_num
        if self.thread_num <= 1:
            self.thread_num = max(os.cpu_count() / 4, 2)
            param.eval_thread_num = self.thread_num
        
        self.datasetfun = datasetfun
        self.modelfun = modelfun
        self.accuracyfun = accuracyfun
        # start a new process, this Can't pickle lambda
        # multiprocessing.set_start_method('spawn')
        self.runner = EvaluateRunner(self, param)
    
    def EvaluateHook(self, every_n_steps=1000):
        return EvaluateHook(self, every_n_steps)
    
    def setup_config(self):
        #for ROC refer: https://blog.csdn.net/jiangjieqazwsx/article/details/52262389
        os.environ['CUDA_VISIBLE_DEVICES']=''

        self.config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        self.config.gpu_options.visible_device_list = ""
        # see ./tensorflow/docs_src/performance/performance_guide.md 
        # Nodes that can use multiple threads to
        # parallelize their execution will schedule the individual pieces into this pool.
        self.config.intra_op_parallelism_threads = self.thread_num
        #All ready nodes are scheduled in this pool.
        self.config.inter_op_parallelism_threads = self.thread_num
        self.config.device_count['CPU'] = self.thread_num
        self.config.device_count['GPU'] = 0 

        self.config.use_per_session_threads = True
        self.config.isolate_session_state = True
        
    
    def setup_graph(self):
        '''
        this run in child process, copy and write willbe deadlock~
        '''
        # can't unimport ~
        import tensorflow as tf
        #clear and build graph
        tf.reset_default_graph()
        self.eval_graph = tf.Graph()
        with self.eval_graph.as_default() as g:
            self.dataset = self.datasetfun(self.param)

            self.iterator = self.dataset.make_initializable_iterator()
            image, label = self.iterator.get_next()
            
            prediction = self.modelfun(image)
            probs = tf.nn.softmax(prediction)
            self.shape = tf.shape(probs)
            self.top1= self.accuracyfun(label, probs)
    
    def post_step(self, step):
        self.runner.queue.put(step)
        
    def setup(self):
        self.setup_config()
        self.setup_graph()
        logging.info("Setup Evaluater done thread_num:"+str(self.thread_num))
    
    def do(self, step):
        
        self.param.model_init.pre_train_dir = self.param.checkpoint
        self.param.model_init.pre_train_step = step

        report = reporter(self.param.model_init.pre_train_dir, step,self.param.validation_num)
        # start e loop
        with self.eval_graph.as_default() as g:
            scaffold = InitScaffold(self.param)            

            with tf.train.MonitoredTrainingSession(scaffold=scaffold, config=self.config) as mon_sess:

                mon_sess.run(self.iterator.initializer)

                try:
                    while not mon_sess.should_stop():                        
                        n1, shape = mon_sess.run([self.top1, self.shape])
                        report.update(n1, shape[0])                                     

                except tf.errors.OutOfRangeError:
                    report.end()  