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


class dA(object):
    """Denoising Auto-Encoder class (dA)"""

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=500, n_hidden=500,
                 W=None, bhid=None, bvis=None, sparsity = None,
                 sparsity_weight = None,
                 hidden_activation = T.nnet.sigmoid,
                 reconstruct_activation = T.nnet.sigmoid):

        self.type = 'fc'

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.sparsity = sparsity
        self.sparsity_weight = sparsity_weight

        self.hidden_activation = hidden_activation
        self.reconstruct_activation = reconstruct_activation
        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng 

        # if no input is given, generate a variable representing the input
        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            if hidden_activation == T.nnet.sigmoid:
                initial_W *= 4
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX),borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), name='b', borrow=True)

        self.W = W
        # b -- the bias of the hidden
        self.b = bhid
        # b_prime -- the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

        self.delta_W = theano.shared(value=numpy.zeros_like(W.get_value(borrow=True), dtype=theano.config.floatX), name='delta_W')
        self.delta_b = theano.shared(value=numpy.zeros_like(bhid.get_value(borrow=True), dtype=theano.config.floatX), name='delta_b')
        self.delta_b_prime = theano.shared(value=numpy.zeros_like(bvis.get_value(borrow=True), dtype=theano.config.floatX), name='delta_b_prime')

        self.params = [self.W, self.b, self.b_prime]
        self.delta_params = [self.delta_W, self.delta_b, self.delta_b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return self.hidden_activation(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the hidden layer """
        return  self.reconstruct_activation(T.dot(hidden, self.W_prime) + self.b_prime)

    def kl_divergence(self, p, p_hat):
        return p * T.log(p / p_hat) + (1 - p) * T.log((1 - p) / (1 - p_hat))

    def get_cost_updates(self, corruption_level, learning_rate, momentum):
        """ This function computes the cost and the updates for one trainng step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        if self.reconstruct_activation is T.tanh:
            L = T.sqr(self.x - z).sum(axis=1)

        if self.sparsity_weight is not None:
            sparsity_level = T.extra_ops.repeat(self.sparsity, self.n_hidden)
            avg_act = y.mean(axis=0)

            kl_div = self.kl_divergence(sparsity_level, avg_act)

            cost = T.mean(L) + self.sparsity_weight * kl_div.sum()
        else:
            cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam*learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        return (cost, updates)

class dA_maxout(dA):
    """Denoising Auto-Encoder with Maxout hidden activation"""

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=500, n_hidden=500,
                 W=None, bhid=None, bvis=None, sparsity = None,
                 sparsity_weight = None,
                 reconstruct_activation = (lambda x: 1.0*x),
                 pool_size = 3):
        super(dA_maxout, self).__init__(numpy_rng=numpy_rng, theano_rng=theano_rng,
                 input=input, n_visible=n_visible, n_hidden = n_hidden,
                 W= W, bhid = bhid, bvis = bvis, sparsity = sparsity, 
                 sparsity_weight = sparsity_weight,
                 hidden_activation = (lambda x: 1.0*x),
                 reconstruct_activation = reconstruct_activation)

        self.pool_size = pool_size
        initial_W_prime = numpy.asarray(numpy_rng.uniform(
                  low=-numpy.sqrt(6. / (n_hidden/pool_size + n_visible)),
                  high=numpy.sqrt(6. / (n_hidden/pool_size + n_visible)),
                  size=(n_hidden/pool_size, n_visible)), dtype=theano.config.floatX)
        W_prime = theano.shared(value=initial_W_prime, name='W_prime', borrow=True)

        # tied weights, therefore W_prime is W transpose
        self.W_prime = W_prime
        self.delta_W_prime = theano.shared(value=numpy.zeros_like(W_prime.get_value(borrow=True), dtype=theano.config.floatX),
                                                                  name='delta_W_prime')

        self.params = [self.W, self.W_prime, self.b, self.b_prime]
        self.delta_params = [self.delta_W, self.delta_W_prime, self.delta_bvis, self.delta_bhid]        

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        lin_output = T.dot(input, self.W) + self.b
        last_start = self.n_hidden - self.pool_size
        tmp_output = lin_output[:,0:last_start+1:self.pool_size]
        for i in range(1, self.pool_size):
            cur = lin_output[:,i:last_start+i+1:self.pool_size]
            tmp_output = T.maximum(cur, tmp_output)

        return tmp_output

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the hidden layer """
        if self.reconstruct_activation is None:
            return  T.dot(hidden, self.W_prime) + self.b_prime
        else:
            return  self.reconstruct_activation(T.dot(hidden, self.W_prime) + self.b_prime)


