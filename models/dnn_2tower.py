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

from dnn import DNN

class DNN_2Tower(object):

    def __init__(self, numpy_rng, theano_rng=None, cfg = None, cfg_tower1 = None, cfg_tower2 = None):

        self.layers = []
        self.params = []
        self.delta_params   = []

        self.cfg = cfg
        self.cfg_tower1 = cfg_tower1
        self.cfg_tower2 = cfg_tower2

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y') 
       
        self.input_tower1 = self.x[:,0:cfg_tower1.n_ins]
        self.input_tower2 = self.x[:,cfg_tower1.n_ins:(cfg_tower1.n_ins+cfg_tower2.n_ins)]
 
        self.dnn_tower1 = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = self.cfg_tower1,
                              input  = self.input_tower1)
        self.dnn_tower2 = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = self.cfg_tower2,
                              input  = self.input_tower2)
        concat_output = T.concatenate([self.dnn_tower1.layers[-1].output, self.dnn_tower2.layers[-1].output], axis=1)
        self.dnn = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = self.cfg, input  = concat_output)

        self.layers.extend(self.dnn_tower1.layers); self.params.extend(self.dnn_tower1.params); 
        self.delta_params.extend(self.dnn_tower1.delta_params)
        self.layers.extend(self.dnn_tower2.layers); self.params.extend(self.dnn_tower2.params);
        self.delta_params.extend(self.dnn_tower2.delta_params)
        self.layers.extend(self.dnn.layers); self.params.extend(self.dnn.params);
        self.delta_params.extend(self.dnn.delta_params)

        self.finetune_cost = self.dnn.logLayer.negative_log_likelihood(self.y)
        self.errors = self.dnn.logLayer.errors(self.y)

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

