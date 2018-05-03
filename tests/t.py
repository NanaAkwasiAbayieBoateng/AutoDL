
import tensorflow as tf
import itertools

from tensorflow.python.framework import tensor_util


def test_gen():
    def gen():
        for i in itertools.count(1):
            yield (i, [1] * i)

    ds = tf.data.Dataset.from_generator(
        # gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))
        gen,
        (tf.int64, tf.int64))

    value = ds.make_one_shot_iterator().get_next()

    sess = tf.Session()
    sess.run(value)  # (1, array([1]))
    sess.run(value)  # (2, array([1, 1]))


if __name__ == '__main__':
    test_gen()
