

"""A simple script for inspect checkpoint files.
   usage : python3.5 inspect_graph_shape.py --file_name=../test/optime_deepsmart.pb --input_shape=1,159,159,3
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import re
import random
import numpy as np
import tensorflow as tf
import logging 

FORMAT = '%(asctime)-15s %(levelname)s %(process)d-%(thread)d %(message)s %(filename)s:%(module)s:%(funcName)s:%(lineno)d'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, handlers=[logging.StreamHandler(),logging.FileHandler(filename=sys.argv[0]+".log", mode='w')])


from google.protobuf import text_format

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import op_def_registry as op_reg
from tensorflow.python.framework import tensor_util


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("file_name", "", "Checkpoint filename")
tf.app.flags.DEFINE_string("input_shape", None, "shappe of the tensor to inspect")

sys.path.append('.')
from graphviz_visual import graphviz_visual
import restore_graph_from_checkpoint as restore

def save_graph_visual(graph_def, name):    
    gv_nodes = {}
    gv_edges = {}
    
    namespace_group = {}
   # for name space 
    for node in graph_def.node:
         node_namespace = node.name.split('/')[0]
         gv_nodes[node_namespace] = (node_namespace, {'label': node_namespace + ":" + repr(node.op)})
         namespace_group[node_namespace] = [node]+ namespace_group.get(node_namespace, [])
         for input_name in node.input:
             edges_start   = input_name.split('/')[0] 
             edges_name    = edges_start + "->"+   node_namespace          
             gv_edges[edges_name] = ((edges_start, node_namespace), {'label': edges_name})
    
    logging.info("save to "+ "namespace_" + name)
    graphviz_visual(gv_nodes.values(), gv_edges.values(), "namespace+" + name)
    
    #for every namespace :
    '''
    for k,v in namespace_group.items():
        gv_nodes = []
        gv_edges = []     
        for n in v:
            gv_nodes += [(n.name, {'label': n.name + "(%s)" % n.op})]        
            #gv_edges += [((i, n.name), {'label': i+ "->" + n.name}) for i in n.input ]
            gv_edges += [(i, n.name) for i in n.input ]
        logging.info("save to "+k +"_"+name ) 
        graphviz_visual(gv_nodes, gv_edges, k +"_"+name)
     '''

def getStartTensor(graph):
    '''
       sess.graph.get_tensor_by_name('shuffle_batch:0')
    '''
    p = None
    for o in graph.get_operations():
        if o.type == 'Placeholder':
            p = o

    result = []
    for o in graph.get_operations():
        for out in o.inputs:
            if out.name.startswith(p.name):
                result += [out]
    return result


def getEndOp(graph):
    all_used = {i.name :(o,0) for o in graph.get_operations() for i in o.inputs }
    no_used = [ o for o in graph.get_operations() for i in o.outputs if i.name not in all_used ]
    
    print("noinput"+repr([i.name for i in no_used]))
    return no_used

def getOperLabel(sess, op):
    label = op.type+":"+ op.name
    label = label +"\ninput:" +','.join([i.name for i in op.inputs])
    label = label +"\noutput:" +','.join([i.name for i in op.outputs])
    
    if op.type == 'Const':
        tensor = op.node_def.attr['value'].tensor
        tensor_value = tensor_util.MakeNdarray(tensor)
        #print(repr(dir(tensor_value)) +","+repr(type(tensor_value)))
    
    if op.type != 'Const':
        label = label +"\nattr:"+ repr([ str(k).replace('\n','') for k in op.node_def.attr.values()])
    #if op.type == 'Conv2D':
        #print(repr([ k.__getattribute__(k.WhichOneof('value')) for k in op.node_def.attr.values()]))
    return label

def getValueLabel(value):
    shape = ""
    val   = None
    if value.shape == ():
        shape = "scalar"
        val = value[()]
    else:
        shape = 'shape:'+','.join([str(i) for i in value.shape])
        val  = [ round(f, 2) for f in  value.flatten().tolist()[:5] ]
        val[-1] = '...' if value.size > 5 else val[-1]
    return shape +"\nvalue:" + str(val)


def getShapeMap(sess, starts, shape_map):
    run_options = tf.RunOptions()
    run_metadata = tf.RunMetadata()

    for o in sess.graph.get_operations():
        if o.type == 'Placeholder':
            continue

        for out in o.outputs:
            if out.name in shape_map:
               continue

            if o.type == 'Const':
               value   = out.eval(session=sess)
               shape_map[out.name] = getValueLabel(value)
               continue

            result =  sess.run(out, starts, run_options, run_metadata)
            starts[out.name] = result
            if isinstance(result, np.ndarray):
                shape_map[out.name] = getValueLabel(result)
            elif isinstance(result, np.float32):
                shape_map[out.name] = "scalar:" + repr(result)
            else :
                shape_map[out.name] = repr(out.get_shape()) + "," +repr(type(result))
                print("Not array:" + repr((o.name, o.type, out.name, out.get_shape(), type(result))))

    return shape_map

def infernce_graph_shape(graph_def_f, shape_str = None):
    graph_name = graph_def_f.split('/')[-1]
   
    graph = restore.load_graph_def(graph_def_f)
    save_graph_visual(graph, graph_name)
    
    if not shape_str:
       return graph
   
    with tf.Session() as sess:
        tensor_shape = [int(i) for i in shape_str.split(',')]
        tensor_value = np.random.rand(*tuple(tensor_shape))
      
        input_tensor = getStartTensor(sess.graph)
        for i in input_tensor:
           i.set_shape(tensor_shape)
        starts = {i : tensor_value for i in input_tensor}
        shape_map =  {i.name : shape_str for i in input_tensor}
        shape_map = getShapeMap(sess, starts, shape_map)

        gv_nodes = []
        gv_edges = []     
        for o in sess.graph.get_operations():
           gv_nodes += [(o.name, {'label':getOperLabel(sess, o)})]
           gv_edges += [((i.op.name, o.name), {'label': str(shape_map.get(i.name, [i.name]))}) for i in o.inputs ]
    
        graphviz_visual(gv_nodes, gv_edges, "shape_" + graph_name)
    return None

def getTensorValue(sess,  out, out_map, value_map):
    
    if out.name in value_map:
       return value_map[out.name]

    op = out_map[out.name]
    
    if op.type == 'Const':
       value_map[out.name] = (True, out.eval(session=sess))
       return (True, value_map[out.name])
    
    if op.type == 'Placeholder':
       value_map[out.name] = (False, None)
       return value_map[out.name]
    
    not_yet = []
    for i in op.inputs:
        if not (i in value_map):
            k,v = getTensorValue(sess, i , out_map, value_map)
            if not k: 
               not_yet.append(i.name )
    
    #not_yet = list(filter(lambda x: value_map[x][0] == True, [x.name for x in op.inputs]))
    
    if len(not_yet) == 0:
        v = sess.run(out, {})
        value_map[out.name] = (True, v)
    else:
        value_map[out.name] = (False, None)
        #print(repr([i for i in not_yet]))
        print(repr(["TO "] + [op.type] + [i.name + repr(value_map[i.name][0]) for i in op.inputs ] + ['->'] + [out.name]) )
        
    return value_map[out.name]

def run(graph_def_f, start=None, end=None):

    graph_name = graph_def_f.split('/')[-1]
   
    restore.load_graph_def(graph_def_f)
    
    with tf.Session() as sess:
        graph = sess.graph
       
        input_map = {i.name :o for o in graph.get_operations() for i in o.inputs }
        out_map   = {i.name :o for o in graph.get_operations() for i in o.outputs }
        value_map = {}
        input_tensor = getStartTensor(sess.graph) if not start else start
        out_op    = getEndOp(sess.graph) if not end else end
      
        v = getTensorValue(sess, out_op[0].outputs[0], out_map, value_map)

def main(unused_argv):
    if not FLAGS.file_name:
        print("Usage: %s  --file_name=graph_def_file_name [--input_shape=1,159,159,3]" % sys.argv[0])
        sys.exit(1)
    infernce_graph_shape(FLAGS.file_name, FLAGS.input_shape)

if __name__ == "__main__":
    tf.app.run()
