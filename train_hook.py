import os
import time
import yaml
import subprocess

from datetime import datetime

#import logging
#logging.basicConfig(level = logging.INFO,format = '%(asctime)s:%(levelname)s - %(message)s')

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks

from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache

from device_manager import gpu


class TrainStateHook(session_run_hook.SessionRunHook):
    '''
    Loger train state
    '''

    def __init__(self, param, lr, total_loss, metrics, every_sec=3):

        self.minbatch = param.minibatch
        self.step_per_epoch = param.step_per_epoch
        self.all_step  = param.all_step
        self.worker_num = gpu.get_nr_gpu()
        self.lr = lr
        self.total_loss = total_loss
        self.every_sec = every_sec
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
        
        
        elapsed_time = time.time() - self._last_time
        elapsed_steps = global_step - self._last_steps

        if elapsed_time < self.every_sec:
            return

        #updater
        self._last_time, self._last_steps = time.time(), global_step

        #steps_per_sec  = elapsed_steps / elapsed_time
        sample_per_sec=  self.minbatch * elapsed_steps / elapsed_time

        self.logging(results, global_step, sample_per_sec)

    def logging(self, results, global_step, sample_per_sec):
        
        epoch = global_step // self.step_per_epoch
        step  = global_step % self.step_per_epoch

        lr = results['self_lr']
        total_loss = results['self_total_loss']
        progress = 100.0 * step / self.step_per_epoch

        formats_str = '''Epoch:{epoch}, step:{step}({progress:.2f}%), samples/sec:{sample:.2f}, total_loss:{total_loss:.4f}, lr:{lr}'''
        formats_str = formats_str.format(
            epoch=epoch,
            step=global_step,
            progress=progress,
            sample=sample_per_sec,
            total_loss=total_loss,
            lr=lr)
        
        metrics = [
            " %s:%s" % (k, v) for k, v in results.items()
            if not k.startswith('self_')
        ]

        tf.logging.info(formats_str + ",".join(metrics))


class SaverHook(tf.train.SessionRunHook):
    def __init__(self,
                 param,
                 save_every_n_steps,
                 max_to_keep=10,
                 saver=None,
                 evaluater=None):
        self._saver = saver if saver is not None else tf.train.Saver(
            max_to_keep=max_to_keep)
        
        self.evaluater = evaluater
        self.param = param
        self._save_path = param.checkpoint
        self._save_mode_path = param.checkpoint + "/model.ckpt"
        self._save_every_n_steps = save_every_n_steps
        self.max_to_keep = max_to_keep
        self._save_list = []

    def after_create_session(self, session, coord):
        '''the graph is finalized and ops can no longer be added to the graph.
        '''
        # save graph 
        with open(self._save_path+"/graph.pb.txt", 'w') as f:
            f.write(str(session.graph.as_graph_def()))
                # save param
        param_str = yaml.dump(self.param, default_flow_style=False)
        with open(self._save_path+"/param.yaml", 'w') as f:
            f.write(param_str)

        logging.info("save train param:"+ repr(self.param))
        return super().after_create_session(session, coord)
        
    def begin(self):
        self._g_step = tf.train.get_global_step()


    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs([self._g_step])

    def after_run(self, run_context, run_values):
        step = run_values.results[0]
        if step != 0 and step % self._save_every_n_steps == 0:
            self._saver.save(
                run_context.session,
                self._save_mode_path,
                global_step=step,
                write_meta_graph=False)
            logging.info('Save checkpoint at globall_step:{1}'.format(
                datetime.now(), step))
            if self.evaluater:
                self.evaluater.post_step(step)
            
            # keep last
            self._save_list.append(step)
            if len(self._save_list) >= self.max_to_keep:
                last = self._save_list[0]
                self._save_list = self._save_list[1:]
                file = "model.ckpt-%s.data-00000-of-00001" %  last
                if os.path.exists(file):
                    os.remove(file)
                file = "model.ckpt-%s.index" %  last
                if os.path.exists(file):
                    os.remove(file)

    def end(self, session):
        step = session.run(tf.train.get_global_step())
        self._saver.save(
            session, self._save_path, global_step=step, write_meta_graph=False)




class ProfilerHook(tf.train.SessionRunHook):
    """
    copy from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/training/basic_session_run_hooks.py
    Captures CPU/GPU profiling information every N steps or seconds.
    This produces files called "timeline-<step>.json", which are in Chrome
    Trace format.
    For more information see:
    https://github.com/catapult-project/catapult/blob/master/tracing/README.md
    """

    def __init__(self,
                 save_steps=None,
                 save_secs=None,
                 output_dir=".",
                 show_dataflow=True,
                 show_memory=False):
        """Initializes a hook that takes periodic profiling snapshots.
    `options.run_metadata` argument of `tf.Session.Run` is used to collect
    metadata about execution. This hook sets the metadata and dumps it in Chrome
    Trace format.
    Args:
      save_steps: `int`, save profile traces every N steps. Exactly one of
          `save_secs` and `save_steps` should be set.
      save_secs: `int` or `float`, save profile traces every N seconds.
      output_dir: `string`, the directory to save the profile traces to.
          Defaults to the current directory.
      show_dataflow: `bool`, if True, add flow events to the trace connecting
          producers and consumers of tensors.
      show_memory: `bool`, if True, add object snapshot events to the trace
          showing the sizes and lifetimes of tensors.
    """
        self._output_file = output_dir+"/timeline-{}.json"
        self._file_writer = SummaryWriterCache.get(output_dir)
        self._show_dataflow = show_dataflow
        self._show_memory = show_memory
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        self._next_step = None
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use ProfilerHook.")


    def before_run(self, run_context):
        self._request_summary = (self._next_step is None
                                 or self._timer.should_trigger_for_step(
                                     self._next_step))
        requests = {"global_step": self._global_step_tensor}
        opts = (
            config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
            if self._request_summary else None)

        return SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._timer.update_last_triggered_step(global_step)
            self._save(global_step, self._output_file.format(global_step),
                       run_values.run_metadata.step_stats)
            self._file_writer.add_run_metadata(run_values.run_metadata,
                                               "step_%d" % global_step)

        self._next_step = global_step + 1

    def _save(self, step, save_path, step_stats):
        logging.info("Saving timeline for %d into '%s'.", step, save_path)
        with gfile.Open(save_path, "w") as f:
            trace = timeline.Timeline(step_stats)
            f.write(
                trace.generate_chrome_trace_format(
                    show_dataflow=self._show_dataflow,
                    show_memory=self._show_memory))



class SummaryHook(tf.train.SessionRunHook):
    '''
    SummaryHook can not go with ProfilerHook
    '''
    def __init__(self, path):
        self.save_path = path

    def begin(self):
        self.merged_ops = tf.summary.merge_all()        
        self._global_step_tensor = training_util._get_or_create_global_step_read() 
        
    def start_tensorboard(self):
        cmd = "CUDA_VISIBLE_DEVICES='' tensorboard --logdir=. --port=8081"
        self.process = subprocess.Popen(cmd, shell=True,cwd=self.save_path)


    def after_create_session(self, session, coord):  
        self.summary_writer = tf.summary.FileWriter(self.save_path, session.graph)
        # TODO start a new process       
        return super().after_create_session(session, coord)

    def before_run(self, run_context): 
        fetches = {'summary': self.merged_ops, 'gloal_step': self._global_step_tensor}     
        return basic_session_run_hooks.SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        summary_val = run_values.results["summary"]
        gloal_step  = run_values.results["gloal_step"]
        self.summary_writer.add_summary(summary_val, gloal_step)
    
    def end(self):
        self.summary_writer.close()
        self.process.terminate()
        self.process.wait(3)
