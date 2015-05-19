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
import json

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer

from layers.conv import ConvLayer, ConvLayerForward
from dnn import DNN

from io_func import smart_open

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

    def __init__(self, numpy_rng, theano_rng=None, cfg = None, testing = False, input = None):

        self.layers = []
        self.params = []
        self.delta_params   = []

        self.conv_layers = []

        self.cfg = cfg
        self.conv_layer_configs = cfg.conv_layer_configs
        self.conv_activation = cfg.conv_activation
        self.use_fast = cfg.use_fast

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        if input == None:
            self.x = T.matrix('x')
        else:
            self.x = input
        self.y = T.ivector('y')

        self.conv_layer_num = len(self.conv_layer_configs)
        for i in xrange(self.conv_layer_num):
            if i == 0:
                input = self.x
            else:
                input = self.layers[-1].output
            config = self.conv_layer_configs[i]
            conv_layer = ConvLayer(numpy_rng=numpy_rng, input=input,
			input_shape = config['input_shape'], filter_shape = config['filter_shape'], poolsize = config['poolsize'],
			activation = self.conv_activation, flatten = config['flatten'], use_fast = self.use_fast, testing = testing)
	    self.layers.append(conv_layer)
            self.conv_layers.append(conv_layer)
	    self.params.extend(conv_layer.params)
            self.delta_params.extend(conv_layer.delta_params)

        self.conv_output_dim = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
        cfg.n_ins = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]

        self.fc_dnn = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = self.cfg, input  = self.layers[-1].output)

        self.layers.extend(self.fc_dnn.layers)
        self.params.extend(self.fc_dnn.params)
        self.delta_params.extend(self.fc_dnn.delta_params)

        self.finetune_cost = self.fc_dnn.logLayer.negative_log_likelihood(self.y)
        self.errors = self.fc_dnn.logLayer.errors(self.y)

    def kl_divergence(self, p, p_hat):
        return p * T.log(p / p_hat) + (1 - p) * T.log((1 - p) / (1 - p_hat))

    # output conv config to files
    def write_conv_config(self, file_path_prefix):
        for i in xrange(len(self.conv_layer_configs)):
            self.conv_layer_configs[i]['activation'] = self.cfg.conv_activation_text
            with smart_open(file_path_prefix + '.' + str(i), 'wb') as fp:
                json.dump(self.conv_layer_configs[i], fp, indent=2, sort_keys = True)
                fp.flush()

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

    def build_extract_feat_function(self, output_layer):

        feat = T.matrix('feat')
        out_da = theano.function([feat], self.layers[output_layer].output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da

class CNN_Forward(object):

    def __init__(self, numpy_rng = None, theano_rng=None, conv_layer_configs = [], non_maximum_erasing = False, use_fast = False):

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
                        activation = config['activation'], flatten = config['flatten'], use_fast = use_fast)
            self.conv_layers.append(conv_layer)

    def build_out_function(self):

        feat = T.tensor4('feat')
        out_da = theano.function([feat], self.conv_layers[-1].output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da
