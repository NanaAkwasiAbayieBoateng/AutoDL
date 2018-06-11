import sys
import os

import tensorflow as tf
import logging

"""
# model config
model_init:
  pre_train_dir: checkpint
  pre_train_step: -1
"""


    

def InitScaffold(params):
    # 
    model_init = params.model_init
    if not model_init.pre_train_dir or len(model_init.pre_train_dir) < 0:
        logging.info("No pre_train_dir found, use a new Scaffold")
        return tf.train.Scaffold()
    
    # see https://docs.python.org/3/library/exceptions.html
    if not os.path.exists(model_init.pre_train_dir):
        raise FileExistsError(model_init.pre_train_dir)

    ckpt_file_or_dir = model_init.pre_train_dir
    if os.path.isdir(model_init.pre_train_dir) and model_init.pre_train_step > 0:
        ckpt_file_or_dir +=  '/model.ckpt-{0}'.format(model_init.pre_train_step)
        if not os.path.exists(ckpt_file_or_dir+".index"):
            raise FileExistsError(ckpt_file_or_dir)
    elif os.path.isdir(model_init.pre_train_dir):
        checkpoint_state = tf.train.get_checkpoint_state(ckpt_file_or_dir)
        if not checkpoint_state:
            logging.info("No " + ckpt_file_or_dir +"/checkppoint found, use a new Scaffold")
            return tf.train.Scaffold()
        
        ckpt_file_or_dir = checkpoint_state.model_checkpoint_path
    
    logging.info("load variable from:" + ckpt_file_or_dir)
    
    reader = tf.train.load_checkpoint(ckpt_file_or_dir)

    # as the model variable maybe has name space
    var_to_shape_map = reader.get_variable_to_shape_map()
    strip_var_to_shape_map = {}
    
    #debug
    #names = [k+":"+str(v) for k,v in var_to_shape_map.items()]
    #print('\n'.join(sorted(names)))


    for k,v in var_to_shape_map.items():
        s = max(k.find('/')+1, 0)
        strip_name = k[s:]
        strip_var_to_shape_map[strip_name]=(v,k)

    # compute the var not exist in checkpoint
    new_vars = []
    restore_vars = {}
    for var in tf.global_variables():
        # op has only one output
        assert var.name.endswith(":0")
        varname = var.name[:-2]

        
        shape = var_to_shape_map.get(varname, None)
        # search in strip map
        if shape == None:
           shape, varname = strip_var_to_shape_map.get(var.op.name, (None,None))
        # 
        if shape == None:
            new_vars.append(var)
            continue
        
        if not var.get_shape().is_compatible_with(shape):
            raise Exception("Variable not match, graph: %s%s, checkpoint:%s%s" %(var.name, var.get_shape().as_list(), varname,shape))
        
        #print("Variable match, graph: %s%s, checkpoint:%s%s" %(var.name, var.get_shape().as_list(), varname,shape))
        #if varname == 'global_step':
        #    print(var)
        restore_vars[varname] = var

    #print("new_vars:"+str(new_vars))

    saver = tf.train.Saver(var_list=restore_vars)

    new_vars_initializer = tf.variables_initializer(new_vars)

    def restore_model(scaffold, session):
        saver.restore(session, ckpt_file_or_dir)

    return tf.train.Scaffold(init_op=new_vars_initializer, init_fn=restore_model, saver=saver)


def test_InitScaffold(params):

    
    from model import official_model
    from data_set      import synthetic_dataset
    
    # dataset 
    ds = synthetic_dataset([4, 3, 32, 32], classnum = 10)
    iterator = ds.make_initializable_iterator()
    image, label = iterator.get_next()

    create_model_func = official_model.Cifar10Model(params.resnet_layer, params.class_num)  


    out = create_model_func(image, training=True);
    global_step = tf.train.get_or_create_global_step()
    
   

    

    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    
    scaffold = InitScaffold(params)
    with tf.train.MonitoredTrainingSession(
        hooks=[], scaffold=scaffold, config=config) as mon_sess:

        mon_sess.run(iterator.initializer)
        #
        mon_sess.run(out)

        step =  mon_sess.run(global_step) 
        print("step:%s" % step)
        

if __name__ == '__main__':
    train_file = './checkpoint/cifar10_train.yaml' if len(sys.argv) == 1 else sys.argv[1]
    import yaml
    from name_dict import NameDict

    print(train_file)
    with open(train_file, 'r') as f:
        params = yaml.load(f)

    test_InitScaffold(params)