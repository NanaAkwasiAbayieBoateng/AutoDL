

## to install sudo pip3 install graphviz
## usage: 
##   graphviz_visual(nodes, egdes, name)
##      gv_nodes += [(node.name, {'label': node.name + ":" + repr(node.op)})]         
##      gv_edges += [((input_name, node.name), {'label': input_name}) for input_name in node.input ]

import graphviz as gv
import functools

import tensorflow as tf
from tensorflow.python.framework import tensor_util
import queue
#### 
styles = {
    'graph': {
        'label': 'A Fancy Graph',
        'fontsize': '16',
        'fontcolor': 'black',
        'bgcolor': '#cccccc',
        'rankdir': 'TB',
        'imagescale':'True',
        'width': '1024',
        'height':'1024',
    },
    'nodes': {
        'fontname': 'Helvetica',
        'shape': 'box',
        'fontcolor': 'white',
        'color': 'white',
        'style': 'filled',
        'fillcolor': '#006699',
        'labeljust': 'l',
    },
    'edges': {
        'style': 'dashed',
        'color': 'green',
        'arrowhead': 'open',
        'fontname': 'Courier',
        'fontsize': '12',
        'fontcolor': 'blue',
    }
}


def add_nodes(graph, nodes):
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph

def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph

def apply_styles(graph, styles):
    graph.graph_attr.update(
        ('graph' in styles and styles['graph']) or {}
    )
    graph.node_attr.update(
        ('nodes' in styles and styles['nodes']) or {}
    )
    graph.edge_attr.update(
        ('edges' in styles and styles['edges']) or {}
    )
    return graph

def graphviz_visual(gv_nodes, gv_edges, imagename):
    digraph = functools.partial(gv.Digraph, format='png')
    # add all edges nodes
    s = set([])
    all_node = {}
    for k in gv_nodes:
       if k[0] not in s:
           s.add(k[0])
           all_node[k[0]] = k
    for e in gv_edges:
       if type(e) != type([]) or len(e) < 2:
           print(e)
           continue
          
       if e[0][0] not in s:
           s.add(e[0][0])
           all_node[e[0][0]] = (e[0][0], {'label': e[0][0]+"(edges)"})
       if e[0][1] not in s:
           s.add(e[0][1])
           all_node[e[0][1]] = (e[0][1], {'label': e[0][1]+"(edges)"})
    
    svg = add_nodes(digraph(), all_node.values())
    svg = add_edges(svg, gv_edges)
    svg = apply_styles(svg, styles)
    svg.render('img/' + imagename)


def print_allchild(tensor_or_op):
    g = tf.get_default_graph()
    input_map = {}
    for op in g.get_operations():
        for i in op.inputs:
           ops = input_map.get(i.name, [])
           if op not in ops:
               input_map[i.name] = ops + [op]
        for i in op.control_inputs:
            ops = input_map.get(i.name, [])
            if op not in ops:
               input_map[i.name] = ops + [op]
    
    if type(tensor_or_op) == tf.Operation:
        op = tensor_or_op
        tensor = tensor_or_op.outputs[0]
    else:
        op = tensor_or_op.op
        tensor = tensor_or_op
    
    q = queue.Queue()
    q.put((0, tensor.name, op))
    print("trace depency: "+tensor.name)
    while not q.empty():
        layer, name, op = q.get()
        ops = input_map.get(name, [])
        if len(ops) == 0:
            continue
            
        print(' '*layer + name + "->"+ ','.join([o.name for o in ops]))
        for o in ops:
           for i in o.outputs:
               q.put((layer+1, i.name, o))



def print_tensortree(tensor_or_op, layer=0, path=['  '], visit=set()):
    '''
    just like pstree, print node layer
    ''' 
    
    def print_path(layer, op, tensor, parenetop, path):
        
        while len(path) <= layer + 1:
            path.append('  ')
        #  set next node prefix 
        path[layer + 1] = '│ ' if len(op.inputs) > 1 else '  '
        
        if len(op.inputs) > 0 and tensor.name == parenetop.inputs[-1].name:
             symbol = "└─┬─"
             #last node clear node prefix
             path[layer] = '  '
        elif len(op.inputs) > 0 and len(parenetop.inputs) == 1:
             symbol = "├───"             
        elif len(op.inputs) > 0:
             symbol =  "├─┬─"
        elif len(parenetop.inputs) > 0 and tensor.name == parenetop.inputs[-1].name:
             symbol = "└───"
        else:
             symbol =  "├───"
        
        label = tensor.name
        if op.type == 'Const':
             tensorobject = op.node_def.attr['value'].tensor
             tensor_value = tensor_util.MakeNdarray(tensorobject)
             label = label + ", %s:%s" % (op.type, tensor_value)
        if op.type != 'Const':
             label = label + ", %s:%s" % (op.type, tensor.get_shape())
             #label = label + ", attr(%s)" % repr([ str(k).replace('\n','') for k in op.node_def.attr.values()])
        label=label.replace('\n', ' ')
        print(''.join(path[:layer]) + symbol + label)
    
    # Compatible operation as tensor 
    if type(tensor_or_op) == tf.Operation and len(tensor_or_op.outputs) > 0:
        op = tensor_or_op
        tensor = tensor_or_op.outputs[0] 
    elif type(tensor_or_op) == tf.Operation:
        op = tensor_or_op
        print("dump:%s" % op.name)
        for i in tensor_or_op.inputs:
           print_tensortree(i)
        return         
    else:
        op = tensor_or_op.op
        tensor = tensor_or_op
    
    if layer == 0:
        print("dump:%s" % op.name)
        print_path(layer, op, tensor, op, path)
        visit.add(tensor.name)
    
    for t in op.inputs:

        next_op = t.op
        print_path(layer + 1, next_op, t, op, path)

        if t.name in visit:
            continue
        
        visit.add(t.name)
        print_tensortree(t, layer + 1, path, visit)
