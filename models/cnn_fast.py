# Copyright 2014    Yajie Miao    Carnegie Mellon University

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
import collections

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer

from layers.conv_fast import ConvLayer

class ConvLayer_Config(object):
    """configuration for a convolutional layer """

    def __init__(self, input_shape=(3,1,28,28), filter_shape=(2, 1, 5, 5), 
                 poolsize=(1, 1), activation=T.tanh, output_shape=(3,1,28,28),
                 flatten = False):
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.poolsize = pool_size
        self.output_shape = output_shape
        self.flatten = flatten

class CNN(object):

    def __init__(self, numpy_rng, theano_rng=None,
                 batch_size = 256, n_outs=500,
		 sparsity = None, sparsity_weight = None, sparse_layer = 3,
                 conv_layer_configs = [],
                 hidden_layers_sizes=[500, 500],
                 conv_activation = T.nnet.sigmoid,
                 full_activation = T.nnet.sigmoid,
                 use_fast = False):

        self.layers = []
        self.params = []
        self.delta_params   = []

        self.sparsity = sparsity
        self.sparsity_weight = sparsity_weight
        self.sparse_layer = sparse_layer

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  
        self.y = T.ivector('y') 
        
        self.conv_layer_num = len(conv_layer_configs)
        self.full_layer_num = len(hidden_layers_sizes)

        for i in xrange(self.conv_layer_num):
            if i == 0:
                input = self.x
                is_input_layer = True
            else:
                input = self.layers[-1].output
                is_input_layer = False
            config = conv_layer_configs[i]
            conv_layer = ConvLayer(numpy_rng=numpy_rng, input=input, is_input_layer = is_input_layer,
			input_shape = config['input_shape'], filter_shape = config['filter_shape'], poolsize = config['poolsize'],
			activation = conv_activation, flatten = config['flatten'])
	    self.layers.append(conv_layer)
	    self.params.extend(conv_layer.params)
            self.delta_params.extend(conv_layer.delta_params)

        self.conv_output_dim = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]

        for i in xrange(self.full_layer_num):
            # construct the sigmoidal layer
            if i == 0:
                input_size = self.conv_output_dim
            else:
                input_size = hidden_layers_sizes[i - 1]
            layer_input = self.layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=full_activation)
            # add the layer to our list of layers
            self.layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            self.delta_params.extend(sigmoid_layer.delta_params)

	# We now need to add a logistic layer on top of the MLP
	self.logLayer = LogisticRegression(
			       input=self.layers[-1].output,
			       n_in=hidden_layers_sizes[-1], n_out=n_outs)
        self.layers.append(self.logLayer)
        self.params.extend(self.logLayer.params)
        self.delta_params.extend(self.logLayer.delta_params)

	if self.sparsity_weight is not None:
            sparsity_level = T.extra_ops.repeat(self.sparsity, 630)
	    avg_act = self.sigmoid_layers[sparse_layer].output.mean(axis=0)
	    kl_div = self.kl_divergence(sparsity_level, avg_act)
	    self.finetune_cost = self.logLayer.negative_log_likelihood(self.y) + self.sparsity_weight * kl_div.sum()     
	else:
            self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        self.errors = self.logLayer.errors(self.y)

    def kl_divergence(self, p, p_hat):
        return p * T.log(p / p_hat) + (1 - p) * T.log((1 - p) / (1 - p_hat))

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam*learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
		self.y: train_set_y[index * batch_size:
			            (index + 1) * batch_size]})

        valid_fn = theano.function(inputs=[index],
              outputs=self.errors,
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
		self.y: valid_set_y[index * batch_size:
			            (index + 1) * batch_size]})

        return train_fn, valid_fn

class CNN_Forward(object):

    def __init__(self, numpy_rng = None, theano_rng=None, conv_layer_configs = [], non_maximum_erasing = False):

        self.conv_layers = []
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.tensor4('x')

        self.conv_layer_num = len(conv_layer_configs)
        for i in xrange(self.conv_layer_num):
            if i == 0:
                input = self.x
            else:
                input = self.conv_layers[-1].output
            config = conv_layer_configs[i]
            conv_layer = ConvLayerForward(numpy_rng=numpy_rng, input = input,
                        filter_shape = config['filter_shape'], poolsize = config['poolsize'],
                        activation = config['activation'], flatten = config['flatten'])
            self.conv_layers.append(conv_layer)


    def build_out_function(self, feat_mats):

        feat = T.tensor4('feat')
        out_da = theano.function([feat], self.conv_layers[-1].output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da
