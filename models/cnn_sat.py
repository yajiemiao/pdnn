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
from cnn import CNN

class CNN_SAT(object):

    def __init__(self, numpy_rng, theano_rng=None, cfg_si = None, cfg_adapt = None, testing = False):

        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        # we assume that i-vectors are appended to speech features in a frame-wise manner  
        cnn_input_shape = cfg_si.conv_layer_configs[0]['input_shape']
        self.feat_dim = cnn_input_shape[-1] * cnn_input_shape[-2] * cnn_input_shape[-3]
        self.ivec_dim = cfg_adapt.n_ins

        self.iv = self.x[:,self.feat_dim:self.feat_dim+self.ivec_dim]
        self.feat = self.x[:,0:self.feat_dim]

        self.params = []
        self.delta_params   = []

        # the i-vector network
        dnn_adapt = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg_adapt, input  = self.iv)
        self.dnn_adapt = dnn_adapt
 
        # the final output layer which has the same dimension as the input features
        linear_func = lambda x: x
        adapt_output_layer = HiddenLayer(rng=numpy_rng,
                                 input=dnn_adapt.layers[-1].output,
                                 n_in=cfg_adapt.hidden_layers_sizes[-1],
                                 n_out=self.feat_dim,
                                 activation=linear_func)
        dnn_adapt.layers.append(adapt_output_layer)
        dnn_adapt.params.extend(adapt_output_layer.params)
        dnn_adapt.delta_params.extend(adapt_output_layer.delta_params)


        # construct the cnn architecture
        cnn_si = CNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg_si, input = self.feat + dnn_adapt.layers[-1].output)
        self.cnn_si = cnn_si

        # construct a function that implements one step of finetunining
        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = cnn_si.fc_dnn.logLayer.negative_log_likelihood(self.y)
        self.errors = cnn_si.fc_dnn.logLayer.errors(self.y)

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

