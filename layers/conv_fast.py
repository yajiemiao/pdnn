# Copyright 2013    Yajie Miao    Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

class ConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, numpy_rng=None, input = None, is_input_layer = False,
                 input_shape=(1,28,28), filter_shape=(2, 1, 5, 5), 
                 poolsize=(1, 1), activation=T.tanh,
                 flatten = False, border_mode = 'valid',
		 non_maximum_erasing = False, W=None, b=None):

        
        assert input_shape[1] == filter_shape[1]
        if is_input_layer:
            self.input = input.reshape(input_shape).dimshuffle(1, 2, 3, 0)
        else:
            self.input = input 
        # Now reconstruct the input_shape and filter_shape
        input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        filter_shape = (filter_shape[1], filter_shape[2], filter_shape[3], filter_shape[0])

        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.poolsize = poolsize

        self.activation = activation
        self.flatten = flatten

        fan_in = numpy.prod(filter_shape[:3])
        fan_out = (filter_shape[3] * numpy.prod(filter_shape[1:3]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
	if W is None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            initial_W = numpy.asarray( numpy_rng.uniform(
                                   low=-W_bound, high=W_bound,
                                   size=filter_shape),
                                   dtype=theano.config.floatX)

            if activation == T.nnet.sigmoid:
                initial_W *= 4
            W = theano.shared(value = initial_W, name = 'W')

        self.W = W
        # the bias is a 1D tensor -- one bias per output feature map
	if b is None:
            b_values = numpy.zeros((filter_shape[3],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
	self.b = b

        # for momentum
	self.delta_W = theano.shared(value = numpy.zeros(filter_shape,
		                     dtype=theano.config.floatX), name='delta_W')

	self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
	                             dtype=theano.config.floatX), name='delta_b')

        # convolve input feature maps with filters
        conv_op = FilterActs()
        contiguous_input = gpu_contiguous(self.input)
        contiguous_filters = gpu_contiguous(self.W)
        conv_out = conv_op(contiguous_input, contiguous_filters)
        y_out = activation(conv_out + self.b.dimshuffle(0, 'x', 'x', 'x'))
        
        pool_op = MaxPool(ds=poolsize[0], stride=poolsize[0])
        pooled_out = pool_op(y_out)
        
	if non_maximum_erasing:
	    ds = tuple(poolsize)
	    po = pooled_out.repeat(ds[0], axis = 2).repeat(ds[1], axis = 3)
	    self.output = T.eq(y_out, po) * y_out
	else:
            self.output = pooled_out

        if flatten:
            self.output = self.output.dimshuffle(3, 0, 1, 2) # c01b to bc01
            self.output = self.output.flatten(2)

        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]


