# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2017 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Inter-process communication using MPI."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader


#from shouter.common import coordinator
from shouter.common import get_ext_suffix

from enum import Enum

#@unique
class OpType(Enum):
    REDUCE     = 1
    ALL_REDUCE = 2
    BROCAST    = 3
    GARTHER    = 4
    ALL_GARTHER= 5

class OpsGraph:
    """ keep and compute the variable dependency"""
    
    
    # class member
    class Variable:
        def __init__(self, tensor, type, dst_ranks=[]):
            self.tensor = tensor
            self.type = type
            #self.src_rank = coordinator.get_rank()
            self.src_rank = 0
            if type in [OpType.ALL_REDUCE, OpType.ALL_GARTHER]:
                #self.dst_ranks = coordinator.get_allranks()
                self.dst_ranks =[]
            else:
                self.dst_ranks = dst_ranks
            
            
    # static member           
    tensors = []

    @classmethod
    def add_allreduce(cls, tensor):
        cls.tensors.append(OpsGraph.Variable(tensor, OpType.ALL_REDUCE))
    
    @classmethod
    def add_reduce(cls, tensor, from_ranks=None):
        cls.tensors.append(OpsGraph.Variable(tensor, OpType.REDUCE, dst_ranks=from_ranks))

    @classmethod
    def add_allgather(cls, tensor):
        cls.tensors.append(OpsGraph.Variable(tensor, OpType.ALL_GARTHER))
    
    @classmethod
    def add_broadcast(cls, tensor, from_ranks=None):
        dst_ranks = from_ranks if from_ranks else coordinator.get_allranks()
        cls.tensors.append(OpsGraph.Variable(tensor, OpType.REDUCE, dst_ranks=from_ranks))
    @classmethod
    def compute_tensorid(cls):
        '''TODO: assign tensorid by session graph ops'''
        coordinator.clear()

        for i, v in enumerate(cls.tensors):
            v.tensorid = i
            coordinator.add_tensortable(i, v.name, v.shape.as_list(), v.type, )

        
        

def _load_library(name, op_list=None):
    """Loads a .so file containing the specified operators.

    Args:
      name: The name of the .so file to load.
      op_list: A list of names of operators that the library should have. If None
          then the .so file's contents will not be verified.

    Raises:
      NameError if one of the required ops is missing.
      NotFoundError if were not able to load .so file.
    """
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    for expected_op in (op_list or []):
        for lib_op in library.OP_LIST.op:
            if lib_op.name == expected_op:
                break
        else:
            raise NameError(
                'Could not find operator %s in dynamic library %s' %
                (expected_op, name))
    return library


SHOUTER_LIB = _load_library('shouter_ops_lib' + get_ext_suffix(),
                        ['ShouterAllgather', 'ShouterAllreduce'])


def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)




def _allreduce(tensor, name=None):
    """An op which sums an input tensor over all the Shouter processes.

    The reduction operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Shouter processes for a given name. The reduction
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, summed across all
      processes.   
    
    """
    if name is None:
        name = 'ShouterAllreduce_%s' % _normalize_name(tensor.name)
    
    global_step = tf.train.get_global_step()
    if not global_step:
        raise NotFoundError("Not found global_step")
    
    OpsGraph.add_allreduce(tensor)
    return SHOUTER_LIB.shouter_allreduce(tensor, global_step, name=name)

ops.NotDifferentiable('ShouterAllreduce')


def allgather(tensor, name=None):
    """An op which concatenates the input tensor with the same input tensor on
    all other Shouter processes.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape, except for the first
    dimension, which is allowed to be different.

    Returns:
      A tensor of the same type as `tensor`, concatenated on dimension zero
      across all processes. The shape is identical to the input shape, except for
      the first dimension, which may be greater and is the sum of all first
      dimensions of the tensors in different Shouter processes.
    """
    if name is None:
        name = 'ShouterAllgather_%s' % _normalize_name(tensor.name)

    OpsGraph.add_allgather(tensor)
    global_step = tf.train.get_global_step()
    return SHOUTER_LIB.shouter_allgather(tensor, global_step, name=name)

ops.NotDifferentiable('ShouterAllgather')

def broadcast(tensor, root_rank, name=None):
    """An op which broadcasts the input tensor on root rank to the same input tensor
    on all other Shouter processes.

    The broadcast operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Shouter processes for a given name. The broadcast
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, with the value broadcasted
      from root rank.
    """
    if name is None:
        name = 'ShouterBroadcast_%s' % _normalize_name(tensor.name)
    OpsGraph.add_broadcast(tensor)

    global_step = tf.train.get_global_step()
    return SHOUTER_LIB.shouter_broadcast(tensor, global_step, name=name, root_rank=root_rank)

ops.NotDifferentiable('ShouterBroadcast')
