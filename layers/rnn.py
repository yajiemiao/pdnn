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

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class RnnLayer(object):

    def __init__(self, rng, input, n_in, n_out, initial_hidden=None, W_rec = None, W=None, b=None,
                 activation=T.tanh, do_maxout = False, pool_size = 1):
        self.input = input
        self.n_in = n_in
        self.n_out = n_out

        self.activation = activation
        self.type = 'rn'
        
        if initial_hidden is None:
            initial_hidden_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            initial_hidden = theano.shared(value=initial_hidden_values, name='h0', borrow=True)
        self.h0 = initial_hidden

        if W_rec is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_out, n_out)), dtype=theano.config.floatX)
            if self.activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W_rec = theano.shared(value=W_values, name='W_rec', borrow=True)

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if self.activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W_rec = W_rec
        self.W = W
        self.b = b


        self.delta_W_rec = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W_rec')
        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')
        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        def one_step(i_t, h_tm1):
            """Perform one step of a simple recurrent network returning the current
            hidden activations and the output.
            `i_t` is the input at the current timestep, `h_tm1` and `o_tm1` are the
            hidden values and outputs of the previous timestep. `h_bias` is the bias
            for the hidden units. `W_in`, `W_out` and `W_rec` are the weight matrices.
            Transfer functions can be specified via `hiddenfunc` and `outfunc` for the
            hidden and the output layer."""
            h_t = self.activation(T.dot(i_t, self.W) + T.dot(h_tm1, self.W_rec) + self.b)
            return h_t

        self.output, _ = theano.scan(fn=one_step, sequences=self.input, outputs_info=self.h0)

        # parameters of the model
        self.params = [self.W_rec, self.W, self.b]
        self.delta_params = [self.delta_W_rec, self.delta_W, self.delta_b]
