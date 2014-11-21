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
import collections

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer, DropoutHiddenLayer, _dropout_from_layer

class DNN_2Tower(object):

    def __init__(self, numpy_rng, theano_rng=None,
                 upper_hidden_layers_sizes=[500, 500], n_outs=10,
                 tower1_hidden_layers_sizes=[500, 500], tower1_n_ins = 100,
                 tower2_hidden_layers_sizes=[500, 500], tower2_n_ins = 100,
                 activation = T.nnet.sigmoid,
                 do_maxout = False, pool_size = 1, 
                 do_pnorm = False, pnorm_order = 1,
                 max_col_norm = None, l1_reg = None, l2_reg = None):

        self.tower1_layers = []
        self.tower2_layers = []
        self.upper_layers = []

        self.params = []
        self.delta_params   = []

        self.max_col_norm = max_col_norm
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x') 
        self.y = T.ivector('y')

        self.tower1_input = self.x[:,0:tower1_n_ins]
        self.tower2_input = self.x[:,tower1_n_ins:(tower1_n_ins + tower2_n_ins)]

        # build tower1
        for i in xrange(len(tower1_hidden_layers_sizes)):
            if i == 0:
                input_size = tower1_n_ins
                layer_input = self.tower1_input
            else:
                input_size = tower1_hidden_layers_sizes[i - 1]
                layer_input = self.tower1_layers[-1].output 

            layer = HiddenLayer(rng=numpy_rng,
                                input=layer_input,
                                n_in=input_size,
                                n_out=tower1_hidden_layers_sizes[i],
                                activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.tower1_layers.append(layer)
            self.params.extend(layer.params)
            self.delta_params.extend(layer.delta_params)

        # build tower2
        for i in xrange(len(tower2_hidden_layers_sizes)):
            if i == 0:
                input_size = tower2_n_ins
                layer_input = self.tower2_input
            else:
                input_size = tower2_hidden_layers_sizes[i - 1]
                layer_input = self.tower2_layers[-1].output     

            layer = HiddenLayer(rng=numpy_rng,
                                input=layer_input,
                                n_in=input_size,
                                n_out=tower2_hidden_layers_sizes[i],
                                activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.tower2_layers.append(layer)
            self.params.extend(layer.params)
            self.delta_params.extend(layer.delta_params)

        for i in xrange(len(upper_hidden_layers_sizes)):
            # construct the sigmoidal layer
            if i == 0:
                input_size = tower1_hidden_layers_sizes[-1] + tower2_hidden_layers_sizes[-1]
                layer_input = T.concatenate([self.tower1_layers[-1].output, self.tower2_layers[-1].output], axis=1)
            else:
                input_size = upper_hidden_layers_sizes[i - 1]
                layer_input = self.upper_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=upper_hidden_layers_sizes[i],
                                        activation=activation)
            # add the layer to our list of layers
            self.upper_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            self.delta_params.extend(sigmoid_layer.delta_params)
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input=self.upper_layers[-1].output,
                         n_in=upper_hidden_layers_sizes[-1], n_out=n_outs)

        self.upper_layers.append(self.logLayer)
        self.params.extend(self.logLayer.params)
        self.delta_params.extend(self.logLayer.delta_params)
       
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

#        if self.l1_reg is not None:
#            for i in xrange(self.n_layers):
#                W = self.params[i * 2]
#                self.finetune_cost += self.l1_reg * (abs(W).sum())

#        if self.l2_reg is not None:
#            for i in xrange(self.n_layers):
#                W = self.params[i * 2]
#                self.finetune_cost += self.l2_reg * T.sqr(W).sum()

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

        if self.max_col_norm is not None:
            for i in xrange(self.n_layers):
                W = self.params[i * 2]
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                    updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

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

