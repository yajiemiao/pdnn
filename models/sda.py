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
from theano.tensor.shared_randomstreams import RandomStreams

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer
from layers.da import dA, dA_maxout

class SdA(object):

    def __init__(self, numpy_rng, theano_rng=None, cfg = None, dnn = None):
        """ Stacked Denoising Autoencoders for DNN Pre-training """

        self.cfg = cfg
        self.hidden_layers_sizes = cfg.hidden_layers_sizes
        self.n_ins = cfg.n_ins
        self.hidden_layers_number = len(self.hidden_layers_sizes)

        self.dA_layers = []

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = dnn.x

        for i in xrange(self.hidden_layers_number):
            # the size of the input is either the number of hidden units of
            # the layer below, or the input size if we are on the first layer
            if i == 0:
                input_size = self.n_ins
                layer_input = self.x
            else:
                input_size = self.hidden_layers_sizes[i - 1]
                layer_input = dnn.layers[i-1].output

            # Construct a denoising autoencoder that shared weights with this layer
            if i == 0:
                reconstruct_activation = cfg.firstlayer_reconstruct_activation
            else:
                reconstruct_activation = cfg.hidden_activation
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=self.hidden_layers_sizes[i],
                          W=dnn.layers[i].W,
                          bhid=dnn.layers[i].b,
                          sparsity = cfg.sparsity,
                          sparsity_weight = cfg.sparsity_weight,
                          hidden_activation = cfg.hidden_activation,
                          reconstruct_activation = reconstruct_activation)
            self.dA_layers.append(dA_layer)

    def pretraining_functions(self, train_set_x, batch_size):

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        momentum = T.scalar('momentum')
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level, learning_rate, momentum)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1),
                              theano.Param(momentum, default=0.5)],
                              outputs=cost,
                              updates=updates,
                              givens={self.x: train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
        return pretrain_fns


# an outdated class for SdA with maxout hidden activation. pre-training has been expirically found to be
# NOT helpful for maxout networks, so we don't update this class
class SdA_maxout(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 corruption_levels=[0.1, 0.1], pool_size = 3,
                 sparsity = None, sparsity_weight = None,
                 first_reconstruct_activation = T.tanh):

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i] * pool_size,
                                        activation=(lambda x: 1.0*x),
                                        do_maxout = True, pool_size = pool_size)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this layer
            if i == 0:
                reconstruct_activation = first_reconstruct_activation
            else:
                reconstruct_activation = (lambda x: 1.0*x)
#               reconstruct_activation = first_reconstruct_activation
            dA_layer = dA_maxout(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i] * pool_size,
                              W=sigmoid_layer.W,
                              bhid=sigmoid_layer.b,
                              sparsity = sparsity,
                              sparsity_weight = sparsity_weight,
                              pool_size = pool_size,
                              reconstruct_activation = reconstruct_activation)
            self.dA_layers.append(dA_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input=self.sigmoid_layers[-1].output,
                         n_in=hidden_layers_sizes[-1], n_out=n_outs)

        self.sigmoid_layers.append(self.logLayer)
        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

    def pretraining_functions(self, train_set_x, batch_size):

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
