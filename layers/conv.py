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

class ConvLayer(object):
    """Convolutional layer with max pooling """

    def __init__(self, numpy_rng=None, input = None, input_shape=(256, 1,28,28), filter_shape=(2, 1, 5, 5), 
                 poolsize=(1, 1), activation=T.tanh,
                 flatten = False, border_mode = 'valid',
		 non_maximum_erasing = False, W=None, b=None,
                 use_fast = False, testing = False):

        self.type = 'conv'

        assert input_shape[1] == filter_shape[1]
        
        if testing:
            self.input = input.reshape((input.shape[0], input_shape[1], input_shape[2], input_shape[3]))
            input_shape = None
        else:
            self.input = input.reshape(input_shape)
        

        self.filter_shape = filter_shape
        self.poolsize = poolsize

        self.activation = activation
        self.flatten = flatten

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
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
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
	self.b = b

        # for momentum
	self.delta_W = theano.shared(value = numpy.zeros(filter_shape,
		                     dtype=theano.config.floatX), name='delta_W')

	self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
	                             dtype=theano.config.floatX), name='delta_b')

        # convolve input feature maps with filters
        if use_fast:
            from theano.sandbox.cuda.basic_ops import gpu_contiguous
            from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
            from pylearn2.sandbox.cuda_convnet.pool import MaxPool

            input_shuffled = self.input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
            filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
            conv_op = FilterActs()
            contiguous_input = gpu_contiguous(input_shuffled)
            contiguous_filters = gpu_contiguous(filters_shuffled)
            conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)
            y_out_shuffled = activation(conv_out_shuffled + self.b.dimshuffle(0, 'x', 'x', 'x'))
            pool_op = MaxPool(ds=poolsize[0], stride=poolsize[0])
            pooled_out = pool_op(y_out_shuffled).dimshuffle(3, 0, 1, 2)
        else:
            conv_out = conv.conv2d(input=self.input, filters=self.W,
                filter_shape=filter_shape, image_shape=input_shape,
                border_mode = border_mode)

            y_out = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            # downsample each feature map individually, using maxpooling
	    pooled_out = downsample.max_pool_2d(input=y_out, ds=poolsize, ignore_border=True)
        
	if non_maximum_erasing:
	    ds = tuple(poolsize)
	    po = pooled_out.repeat(ds[0], axis = 2).repeat(ds[1], axis = 3)
	    self.output = T.eq(y_out, po) * y_out
	else:
            self.output = pooled_out

        if flatten:
            self.output = self.output.flatten(2)

        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]

class ConvLayerForward(object):

    def __init__(self, numpy_rng=None, input = None, filter_shape=(2, 1, 5, 5),
                 poolsize=(1, 1), activation=T.nnet.sigmoid,
                 flatten = False, use_fast = False):

        self.type = 'conv'

        self.input = input

        self.filter_shape = filter_shape
        self.poolsize = poolsize

        self.activation = activation
        self.flatten = flatten

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
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
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b')

        # convolve input feature maps with filters
        if use_fast:
            from theano.sandbox.cuda.basic_ops import gpu_contiguous
            from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
            from pylearn2.sandbox.cuda_convnet.pool import MaxPool
       
            input_shuffled = self.input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
            filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
            conv_op = FilterActs()
            contiguous_input = gpu_contiguous(input_shuffled)
            contiguous_filters = gpu_contiguous(filters_shuffled)
            conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)
            y_out_shuffled = activation(conv_out_shuffled + self.b.dimshuffle(0, 'x', 'x', 'x'))
            pool_op = MaxPool(ds=poolsize[0], stride=poolsize[0])
            self.output = pool_op(y_out_shuffled).dimshuffle(3, 0, 1, 2)
        else: 
            conv_out = conv.conv2d(input=self.input, filters=self.W,
                filter_shape=filter_shape)

            y_out = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            # downsample each feature map individually, using maxpooling
            self.output = downsample.max_pool_2d(input=y_out,
                                             ds=poolsize, ignore_border=True)
        if self.flatten:
            self.output = self.output.flatten(2)

        self.params = [self.W, self.b]

