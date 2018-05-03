import tensorflow as tf

"""
  batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
  Get a learning rate that decays step-wise as training progresses.
"""

def PiecewiseLR(train_samples_num, minibatch_size, initial_learning_rate=0.1, global_step=None):
    # empirical from Master, 
    boundary_epochs = [30, 60, 80, 90]
    decay_rates = [1, 0.1, 0.01, 0.001, 0.0001]
    
    initial_learning_rate = initial_learning_rate * minibatch_size / 256

    one_epoch_step = int(train_samples_num / minibatch_size)

    step_boundaries = [one_epoch_step*epoch for epoch in boundary_epochs]

    vals = [initial_learning_rate * lr for lr in decay_rates]
    
    if not global_step:
        global_step = tf.train.get_or_create_global_step()
    
    global_step = tf.cast(global_step, tf.int32)    
    tf.logging.info("PiecewiseLR, step_boundaries:%s, value:%s" %  (step_boundaries, vals))
    return tf.train.piecewise_constant(global_step, step_boundaries, vals)

def test_PiecewiseLR():
    global_step = tf.train.get_or_create_global_step()
    allsample = 500
    minibatchsize = 128
    lr = PiecewiseLR(allsample, minibatchsize, global_step)
    updater = tf.assign(global_step, global_step + 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(300):
           print(sess.run([global_step, lr]))
           sess.run([updater])

if __name__ == '__main__':
    test_PiecewiseLR()
            
      
