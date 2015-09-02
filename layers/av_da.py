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
import collections

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class av_dA(object):
    """Denoising Auto-Encoder class (dA)"""

    def __init__(self, numpy_rng, theano_rng=None, a_input=None, v_input=None,
                 a_visible=500, v_visible=500, n_hidden=500,
                 a_W=None, v_W=None, a_bhid=None, v_bhid=None, a_bvis=None,v_bvis=None, sparsity = None,
                 sparsity_weight = None,
                 hidden_activation = T.nnet.sigmoid,
                 reconstruct_activation = T.nnet.sigmoid):
        """
        :param numpy_rng:
        :param theano_rng:
        :param input:
        :param a_visible:
        :param v_visible:
        :param a_hidden:
        :param v_hidden:
        :param W:
        :param a_bhid:
        :param v_bhid:
        :param a_bvis:
        :param v_bvis:
        :param sparsity:
        :param sparsity_weight:
        :param hidden_activation: activation function, could be T.nnet.sigmoid(default),T.tanh
        :param reconstruct_activation: activation function, could be T.nnet.sigmoid(default),T.tanh
        :return:
        """

        self.type = 'fc'

        self.n_visible = a_visible+v_visible
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
        if a_input == None:
            self.a_x = T.dmatrix(name='a_input')
        else:
            self.a_x = a_input

        if v_input == None:
            self.v_x = T.dmatrix(name='v_input')
        else:
            self.v_x = v_input

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not a_W:
            initial_a_W = numpy.asarray(numpy_rng.uniform(
                      low=-numpy.sqrt(6. / (n_hidden + a_visible)),
                      high=numpy.sqrt(6. / (n_hidden + a_visible)),
                      size=(a_visible, n_hidden)), dtype=theano.config.floatX)
            if hidden_activation == T.nnet.sigmoid:
                initial_a_W *= 4
            a_W = theano.shared(value=initial_a_W, name='a_W', borrow=True)
        if not v_W:
            initial_v_W = numpy.asarray(numpy_rng.uniform(
                      low=-numpy.sqrt(6. / (n_hidden + v_visible)),
                      high=numpy.sqrt(6. / (n_hidden + v_visible)),
                      size=(v_visible, n_hidden)), dtype=theano.config.floatX)
            if hidden_activation == T.nnet.sigmoid:
                initial_v_W *= 4
            v_W = theano.shared(value=initial_v_W, name='v_W', borrow=True)

        if not a_bvis:
            a_bvis = theano.shared(value=numpy.zeros(a_visible, dtype=theano.config.floatX),borrow=True)
        if not v_bvis:
            v_bvis = theano.shared(value=numpy.zeros(v_visible, dtype=theano.config.floatX),borrow=True)

        if not a_bhid:
            a_bhid = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), name='a_b', borrow=True)
        if not v_bhid:
            v_bhid = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), name='v_b', borrow=True)

        # define parameters(for each modality)
        self.a_W = a_W
        self.v_W = v_W
        # b -- the bias of the hidden
        self.a_b = a_bhid
        self.v_b = v_bhid
        # b_prime -- the bias of the visible
        self.a_b_prime = a_bvis
        self.v_b_prime = v_bvis
        # tied weights, therefore W_prime is W transpose
        self.a_W_prime = self.a_W.T
        self.v_W_prime = self.v_W.T

        self.delta_a_W = theano.shared(value=numpy.zeros_like(a_W.get_value(borrow=True), dtype=theano.config.floatX), name='delta_a_W')
        self.delta_v_W = theano.shared(value=numpy.zeros_like(v_W.get_value(borrow=True), dtype=theano.config.floatX), name='delta_v_W')
        self.delta_a_b = theano.shared(value=numpy.zeros_like(a_bhid.get_value(borrow=True), dtype=theano.config.floatX), name='delta_a_b')
        self.delta_v_b = theano.shared(value=numpy.zeros_like(v_bhid.get_value(borrow=True), dtype=theano.config.floatX), name='delta_v_b')
        self.delta_a_b_prime = theano.shared(value=numpy.zeros_like(a_bvis.get_value(borrow=True), dtype=theano.config.floatX), name='delta_a_b_prime')
        self.delta_v_b_prime = theano.shared(value=numpy.zeros_like(v_bvis.get_value(borrow=True), dtype=theano.config.floatX), name='delta_v_b_prime')

        self.params = [self.a_W, self.v_W, self.a_b, self.v_b, self.a_b_prime, self.v_b_prime]
        self.delta_params = [self.delta_a_W, self.delta_v_W, self.delta_a_b, self.delta_v_b, self.delta_a_b_prime, self.delta_v_b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, a_input, v_input):
        """ Computes the values of the hidden layer """
        return self.hidden_activation(self.hidden_activation(T.dot(a_input,self.a_W)+self.a_b) + self.hidden_activation(T.dot(v_input,self.v_W)+self.v_b))

    def get_reconstructed_input(self, hidden, W_prime, b_prime):
        """Computes the reconstructed input given the values of the hidden layer """
        return  self.reconstruct_activation(T.dot(hidden, W_prime) + b_prime)

    def kl_divergence(self, p, p_hat):
        return p * T.log(p / p_hat) + (1 - p) * T.log((1 - p) / (1 - p_hat))

    def get_cost_updates(self, corruption_level, learning_rate, momentum):
        """ This function computes the cost and the updates for one trainng step of the dA """

        tilde_a_x = self.get_corrupted_input(self.a_x, corruption_level)
        tilde_v_x = self.get_corrupted_input(self.v_x, corruption_level)
        y = self.get_hidden_values(tilde_a_x,tilde_v_x)
        a_z = self.get_reconstructed_input(y,self.a_W_prime,self.a_b_prime)
        v_z = self.get_reconstructed_input(y,self.v_W_prime,self.v_b_prime)

        L = - (T.sum(self.a_x*T.log(a_z)+(1-self.a_x)*T.log(1-a_z),axis=1) + T.sum(self.v_x*T.log(v_z)+(1-self.v_x)*T.log(1-v_z),axis=1))
        if self.reconstruct_activation is T.tanh:
            L = T.sqr(self.a_x-a_z).sum(axis=1) + T.sqr(self.v_x-v_z).sum(axis=1)

        # if self.sparsity_weight is not None:
        #     sparsity_level = T.extra_ops.repeat(self.sparsity, self.n_hidden)
        #     avg_act = y.mean(axis=0)
        #
        #     kl_div = self.kl_divergence(sparsity_level, avg_act)
        #
        #     cost = T.mean(L) + self.sparsity_weight * kl_div.sum()
        # else:
        #     cost = T.mean(L)
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

class av_dA_maxout(av_dA):
    """Denoising Auto-Encoder with Maxout hidden activation"""

    def __init__(self, numpy_rng, theano_rng=None, a_input=None, v_input=None,
                 a_visible=500, v_visible=500, n_hidden=500,
                 a_W=None, v_W=None, a_bhid=None, v_bhid=None, a_bvis=None,v_bvis=None, sparsity = None,
                 sparsity_weight = None,
                 reconstruct_activation = (lambda x: 1.0*x),
                 pool_size = 3):
        super(av_dA_maxout, self).__init__(numpy_rng=numpy_rng, theano_rng=theano_rng,
                 a_input=a_input, v_input=v_input, a_visible=a_visible, v_visible=v_visible, n_hidden = n_hidden,
                 a_W= a_W, v_W= v_W, a_bhid = a_bhid, v_bhid = v_bhid, a_bvis = a_bvis, v_bvis = v_bvis, sparsity = sparsity,
                 sparsity_weight = sparsity_weight,
                 hidden_activation = (lambda x: 1.0*x),
                 reconstruct_activation = reconstruct_activation)

        self.pool_size = pool_size
        initial_a_W_prime = numpy.asarray(numpy_rng.uniform(
                  low=-numpy.sqrt(6. / (n_hidden/pool_size + a_visible)),
                  high=numpy.sqrt(6. / (n_hidden/pool_size + a_visible)),
                  size=(n_hidden/pool_size, a_visible)), dtype=theano.config.floatX)
        a_W_prime = theano.shared(value=initial_a_W_prime, name='a_W_prime', borrow=True)

        initial_v_W_prime = numpy.asarray(numpy_rng.uniform(
                  low=-numpy.sqrt(6. / (n_hidden/pool_size + v_visible)),
                  high=numpy.sqrt(6. / (n_hidden/pool_size + v_visible)),
                  size=(n_hidden/pool_size, v_visible)), dtype=theano.config.floatX)
        v_W_prime = theano.shared(value=initial_v_W_prime, name='v_W_prime', borrow=True)

        # tied weights, therefore W_prime is W transpose
        self.W_prime = a_W_prime
        self.delta_a_W_prime = theano.shared(value=numpy.zeros_like(a_W_prime.get_value(borrow=True), dtype=theano.config.floatX),
                                                                  name='delta_a_W_prime')

        self.v_W_prime = v_W_prime
        self.delta_v_W_prime = theano.shared(value=numpy.zeros_like(v_W_prime.get_value(borrow=True), dtype=theano.config.floatX),
                                                                  name='delta_v_W_prime')

        self.params = [self.a_W, self.v_W, self.a_W_prime, self.v_W_prime, self.a_b, self.v_b, self.a_b_prime, self.v_b_prime]
        self.delta_params = [self.delta_a_W, self.delta_v_W, self.delta_a_W_prime, self.delta_v_W_prime, self.delta_a_bvis, self.delta_v_bvis, self.delta_a_bhid, self.delta_v_bhid]

    def get_hidden_values(self, a_input, v_input):
        """ Computes the values of the hidden layer """
        a_lin_output = T.dot(a_input, self.a_W) + self.a_b
        v_lin_output = T.dot(v_input, self.v_W) + self.v_b
        last_start = self.n_hidden - self.pool_size
        a_tmp_output = a_lin_output[:,0:last_start+1:self.pool_size]
        for i in range(1, self.pool_size):
            cur = a_lin_output[:,i:last_start+i+1:self.pool_size]
            a_tmp_output = T.maximum(cur, a_tmp_output)
        v_tmp_output = v_lin_output[:,0:last_start+1:self.pool_size]
        for i in range(1, self.pool_size):
            cur = v_lin_output[:,i:last_start+i+1:self.pool_size]
            v_tmp_output = T.maximum(cur, v_tmp_output)

        return (a_tmp_output+v_tmp_output)

    def get_reconstructed_input(self, hidden, W_prime, b_prime):
        """Computes the reconstructed input given the values of the hidden layer """
        if self.reconstruct_activation is None:
            return  T.dot(hidden, W_prime) + b_prime
        else:
            return  self.reconstruct_activation(T.dot(hidden, W_prime) + b_prime)
