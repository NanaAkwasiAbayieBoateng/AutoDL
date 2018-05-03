import numpy as np
import tensorflow as tf
import itertools
import six

logging = tf.logging


from tensorflow.python.framework import tensor_util
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    """Hook to print out examples per second.

    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
    """

    def __init__(
            self,
            batch_size,
            every_n_steps=100,
            initer=None,
            enqs=None,
            every_n_secs=None
    ):
        """Initializer for ExamplesPerSecondHook.

      Args:
      batch_size: Total batch size used to calculate examples/second from
      global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds.
    """
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps'
                             ' and every_n_secs should be provided.')
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size
        self.initer = initer
        self.enqs = enqs
        logging.info("ExamplesPerSecondHook create")

    def after_create_session(self, session, coord):
        logging.info("after_create_session")
        
        session.run(self.initer)
        return super().after_create_session(session, coord)

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                'Global step should be created to use StepCounterHook.')
    

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context

        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                global_step)
            if elapsed_time is not None:
                steps_per_sec = elapsed_steps / elapsed_time
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = self._batch_size * (
                    self._total_steps / self._step_train_time)
                current_examples_per_sec = steps_per_sec * self._batch_size
                # Average examples/sec followed by current examples/sec
                logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                             average_examples_per_sec,
                             current_examples_per_sec, self._total_steps)
    def end(self, session): 
        return super().end(session)
