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

import os
import sys

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layers.rbm import RBM, GBRBM

class SRBM(object):

    def __init__(self, numpy_rng, theano_rng=None, cfg=None, dnn=None):
        """ Stacked RBMs for DNN Pre-training """

        self.cfg = cfg
        self.hidden_layers_sizes = cfg.hidden_layers_sizes
        self.n_ins = cfg.n_ins
        self.hidden_layers_number = len(self.hidden_layers_sizes)
        
        self.rbm_layers = []

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
        # share the input of the corresponding dnn
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

            # the first layer could be Gaussian-Bernoulli RBM
            # other layers have to be Bernoulli-Bernoulli RBMs
            if i == 0 and cfg.first_layer_gb:
                rbm_layer = GBRBM(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=self.hidden_layers_sizes[i],
                              W=dnn.layers[i].W,
                              hbias=dnn.layers[i].b)
            else:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=self.hidden_layers_sizes[i],
                              W=dnn.layers[i].W,
                              hbias=dnn.layers[i].b)
            self.rbm_layers.append(rbm_layer)

    def pretraining_functions(self, train_set_x, batch_size, k , weight_cost):

        index = T.lscalar('index')  
        momentum = T.scalar('momentum')
        learning_rate = T.scalar('lr') 
        # number of mini-batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # start and end index of this mini-batch
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            r_cost, fe_cost, updates = rbm.get_cost_updates(batch_size, learning_rate,
                                                            momentum, weight_cost,
                                                            persistent=None, k = k)

            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(learning_rate, default=0.0001),
                              theano.Param(momentum, default=0.5)],
                              outputs= [r_cost, fe_cost],
                              updates=updates,
                              givens={self.x: train_set_x[batch_begin:batch_end]})
            # append function to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

