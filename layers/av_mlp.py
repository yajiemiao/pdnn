# Copyright  2015    Fei Tao

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

class av_HiddenLayer(object):
    def __init__(self, rng, a_input, v_input, a_n_in, v_n_in, n_out, a_W=None, v_W=None, a_b=None, v_b=None,
                 activation=T.tanh, do_maxout = False, pool_size = 1):
        """ Class for hidden layer """
        self.a_input = a_input
        self.v_input = v_input
        self.a_n_in = a_n_in
        self.v_n_in = v_n_in
        self.n_out = n_out

        self.activation = activation

        self.type = 'fc'

        if a_W is None:
            a_W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (a_n_in + n_out)),
                    high=numpy.sqrt(6. / (a_n_in + n_out)),
                    size=(a_n_in, n_out)), dtype=theano.config.floatX)
            if self.activation == theano.tensor.nnet.sigmoid:
                a_W_values *= 4
            a_W = theano.shared(value=a_W_values, name='a_W', borrow=True)

        if v_W is None:
            v_W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (v_n_in + n_out)),
                    high=numpy.sqrt(6. / (v_n_in + n_out)),
                    size=(v_n_in, n_out)), dtype=theano.config.floatX)
            if self.activation == theano.tensor.nnet.sigmoid:
                v_W_values *= 4
            v_W = theano.shared(value=v_W_values, name='v_W', borrow=True)

        if a_b is None:
            a_b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            a_b = theano.shared(value=a_b_values, name='a_b', borrow=True)
        if v_b is None:
            v_b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            v_b = theano.shared(value=v_b_values, name='v_b', borrow=True)

        self.a_W = a_W
        self.a_b = a_b
        self.v_W = v_W
        self.v_b = v_b

        self.delta_a_W = theano.shared(value = numpy.zeros((a_n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_a_W')
        self.delta_a_b = theano.shared(value = numpy.zeros_like(self.a_b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_a_b')
        self.delta_v_W = theano.shared(value = numpy.zeros((v_n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_v_W')
        self.delta_v_b = theano.shared(value = numpy.zeros_like(self.v_b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_v_b')

        a_lin_output = T.dot(a_input, self.a_W) + self.a_b
        v_lin_output = T.dot(v_input, self.v_W) + self.v_b
        if do_maxout == True:
            self.last_start = n_out - pool_size
            self.tmp_output = a_lin_output[:,0:self.last_start+1:pool_size]+v_lin_output[:,0:self.last_start+1:pool_size]
            for i in range(1, pool_size):
                cur = a_lin_output[:,i:self.last_start+i+1:pool_size]+v_lin_output[:,i:self.last_start+i+1:pool_size]
                self.tmp_output = T.maximum(cur, self.tmp_output)
            self.output = self.activation(self.tmp_output)
        else:
            self.output = (a_lin_output+v_lin_output if self.activation is None
                           else self.activation(a_lin_output+v_lin_output))

        # parameters of the model
        self.params = [self.a_W, self.a_b, self.v_W, self.v_b]
        self.delta_params = [self.delta_a_W, self.delta_a_b, self.delta_v_W, self.delta_v_b]

def _dropout_from_layer(theano_rng, hid_out, p):
    """ p is the factor for dropping a unit """
    # p=1-p because 1's indicate keep and p is prob of dropping
    return theano_rng.binomial(n=1, p=1-p, size=hid_out.shape,
                               dtype=theano.config.floatX) * hid_out

class av_DropoutHiddenLayer(av_HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 W=None, b=None, activation=T.tanh, do_maxout = False, pool_size = 1, dropout_factor=0.5):
        super(av_DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, do_maxout = do_maxout, pool_size = pool_size)

        self.theano_rng = RandomStreams(rng.randint(2 ** 30))

        self.dropout_output = _dropout_from_layer(theano_rng = self.theano_rng,
                                                  hid_out = self.output, p=dropout_factor)
