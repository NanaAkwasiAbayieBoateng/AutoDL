import tensorflow as tf

"""
  batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
  Get a learning rate that decays step-wise as training progresses.
"""

def PiecewiseLR(params, global_step=None):
    # empirical from Master, 
    boundary_epochs = params.boundaries
    decay_rates = params.boundaries
    

    initial_learning_rate = params.initial_learning_rate * param.minibatch / params.batch_denom

    batches_per_epoch = params.train_nums / param.minibatch

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]
    
    if not global_step:
        global_step = tf.train.get_or_create_global_step()
        global_step = tf.cast(global_step, tf.int32)
    tf.logging.info("Set PiecewiseLR, step_boundaries:%s, value:%s" %  (boundaries, vals))
    return tf.train.piecewise_constant(global_step, boundaries, vals)

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
            
      
