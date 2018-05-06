import os
import time
from datetime import datetime

#import logging
#logging.basicConfig(level = logging.INFO,format = '%(asctime)s:%(levelname)s - %(message)s')

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks

from device_manager import gpu


class TrainStateHook(session_run_hook.SessionRunHook):
    '''
    Loger train state
    '''
    def __init__(self, param, lr, total_loss, metrics, every_steps=100):
       
        self.minbatch = param.minibatch
        self.worker_num = gpu.get_nr_gpu()
        self.lr = lr
        self.total_loss = total_loss
        self.every_steps = every_steps 
        self.fetches = metrics 
        self._start_steps = None
    
    def begin(self):
        self.global_step = tf.train.get_global_step()
        
        self.fetches['self_global_step'] = self.global_step
        self.fetches['self_lr'] = self.lr 
        self.fetches['self_total_loss'] = self.total_loss 
    
    def end(self, session):
        '''when session.run() raises OutOfRangeError or StopIteration.
         In that case end() is called but after_run() is not called.
        '''
        return super().end(session)

    def after_create_session(self, session, coord):
        '''the graph is finalized and ops can no longer be added to the graph.
        '''
        return super().after_create_session(session, coord)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        # extend session.run(ops) so the ops can be excute parallel
        return basic_session_run_hooks.SessionRunArgs(fetches=self.fetches)

    def after_run(self, run_context, run_values):
        results = run_values.results
        global_step = results['self_global_step']
        if not self._start_steps:
            self._start_steps = global_step
            self._start_time = time.time()
            self._last_time = self._start_time
            self._last_steps = self._start_steps
            return 

        if global_step <= (self._last_steps + self.every_steps):
            return       

        elapsed_time  = time.time() - self._last_time
        elapsed_steps = global_step - self._last_steps
        #updater
        self._last_time, self._last_steps =  time.time(), global_step

        results['self_steps_per_sec']  = elapsed_steps / elapsed_time
        results['self_sample_per_sec'] = results['self_steps_per_sec'] * self.minbatch
        
        self.logging(results)
    
    def logging(self, results):
        global_step = results['self_global_step']
        steps_per_sec = results['self_steps_per_sec']
        sample_per_sec = results['self_sample_per_sec']
        lr = results['self_lr']
        total_loss = results['self_total_loss']

        formats_str= '''global_step:{step}, steps/sec:{avg:.2f}, samples/sec:{sample:.2f}, total_loss:{total_loss:.2f}, lr:{lr}'''
        formats_str= formats_str.format(step=global_step, avg=steps_per_sec, sample=sample_per_sec, total_loss=total_loss, lr=lr)

        metrics = [" %s:%s" %(k,v) for k,v in results.items() if not k.startswith('self_')]

        tf.logging.info(formats_str + ",".join(metrics))

class SaverHook(tf.train.SessionRunHook):
    
    def __init__(self,
                checkpoint_dir,
                save_every_n_steps,
                max_to_keep=100,
                saver=None):
        self._saver = saver if saver is not None else tf.train.Saver(max_to_keep=max_to_keep)
        self._save_path = os.path.join(checkpoint_dir, "model.ckpt")
        self._save_every_n_steps = save_every_n_steps

    def begin(self):
        self._g_step = tf.train.get_global_step()

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs([self._g_step])

    def after_run(self, run_context, run_values):
        step = run_values.results[0]
        if step != 0 and step % self._save_every_n_steps == 0:
            self._saver.save(run_context.session, self._save_path, 
                             global_step=step, write_meta_graph=False)
            logging.info('{0} Save checkpoint at {1}'.format(datetime.now(), step))

    def end(self, session):
        step = session.run(tf.train.get_global_step())
        self._saver.save(session, self._save_path, 
                             global_step=step, write_meta_graph=False)