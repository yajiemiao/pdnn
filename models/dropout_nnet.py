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

from io_func import smart_open
from io_func.model_io import _nnet2file, _file2nnet

class DNN_Dropout(object):

    def __init__(self, numpy_rng, theano_rng=None,
                 cfg = None,
                 dnn_shared = None, shared_layers=[]):

        self.layers = []
        self.dropout_layers = []
        self.params = []
        self.delta_params   = []

        self.cfg = cfg
        self.n_ins = cfg.n_ins; self.n_outs = cfg.n_outs
        self.hidden_layers_sizes = cfg.hidden_layers_sizes
        self.hidden_layers_number = len(self.hidden_layers_sizes)
        self.activation = cfg.activation

        self.do_maxout = cfg.do_maxout; self.pool_size = cfg.pool_size
        self.input_dropout_factor = cfg.input_dropout_factor; self.dropout_factor = cfg.dropout_factor

        self.max_col_norm = cfg.max_col_norm
        self.l1_reg = cfg.l1_reg
        self.l2_reg = cfg.l2_reg

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in xrange(self.hidden_layers_number):
            # construct the hidden layer
            if i == 0:
                input_size = self.n_ins
                layer_input = self.x
                if self.input_dropout_factor > 0.0:
                    dropout_layer_input = _dropout_from_layer(theano_rng, self.x, self.input_dropout_factor)
                else:
                    dropout_layer_input = self.x
            else:
                input_size = self.hidden_layers_sizes[i - 1]
                layer_input = (1 - self.dropout_factor[i - 1]) * self.layers[-1].output
                dropout_layer_input = self.dropout_layers[-1].dropout_output

            W = None; b = None
            if (i in shared_layers) :
                W = dnn_shared.layers[i].W; b = dnn_shared.layers[i].b

            if self.do_maxout == False:
                dropout_layer = DropoutHiddenLayer(rng=numpy_rng,
                                        input=dropout_layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i],
                                        W = W, b = b,
                                        activation= self.activation,
                                        dropout_factor=self.dropout_factor[i])
                hidden_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i],
                                        activation= self.activation,
                                        W=dropout_layer.W, b=dropout_layer.b)
            else:
                dropout_layer = DropoutHiddenLayer(rng=numpy_rng,
                                        input=dropout_layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i] * self.pool_size,
                                        W = W, b = b,
                                        activation= (lambda x: 1.0*x),
                                        dropout_factor=self.dropout_factor[i],
                                        do_maxout = True, pool_size = self.pool_size)
                hidden_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i] * self.pool_size,
                                        activation= (lambda x: 1.0*x),
                                        W=dropout_layer.W, b=dropout_layer.b,
                                        do_maxout = True, pool_size = self.pool_size)
            # add the layer to our list of layers
            self.layers.append(hidden_layer)
            self.dropout_layers.append(dropout_layer)
            self.params.extend(dropout_layer.params)
            self.delta_params.extend(dropout_layer.delta_params)
        # We now need to add a logistic layer on top of the MLP
        self.dropout_logLayer = LogisticRegression(
                                 input=self.dropout_layers[-1].dropout_output,
                                 n_in=self.hidden_layers_sizes[-1], n_out=self.n_outs)

        self.logLayer = LogisticRegression(
                         input=(1 - self.dropout_factor[-1]) * self.layers[-1].output,
                         n_in=self.hidden_layers_sizes[-1], n_out=self.n_outs,
                         W=self.dropout_logLayer.W, b=self.dropout_logLayer.b)

        self.dropout_layers.append(self.dropout_logLayer)
        self.layers.append(self.logLayer)
        self.params.extend(self.dropout_logLayer.params)
        self.delta_params.extend(self.dropout_logLayer.delta_params)

        # compute the cost
        self.finetune_cost = self.dropout_logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

        if self.l1_reg is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                self.finetune_cost += self.l1_reg * (abs(W).sum())

        if self.l2_reg is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                self.finetune_cost += self.l2_reg * T.sqr(W).sum()

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
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
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

    def write_model_to_raw(self, file_path):
        # output the model to tmp_path; this format is readable by PDNN
        _nnet2file(self.layers, filename=file_path, input_factor = self.input_dropout_factor, factor = self.dropout_factor)

    def write_model_to_kaldi(self, file_path, with_softmax = True):

        # determine whether it's BNF based on layer sizes
        output_layer_number = -1;
        for layer_index in range(1, self.hidden_layers_number - 1):
            cur_layer_size = self.hidden_layers_sizes[layer_index]
            prev_layer_size = self.hidden_layers_sizes[layer_index-1]
            next_layer_size = self.hidden_layers_sizes[layer_index+1]
            if cur_layer_size < prev_layer_size and cur_layer_size < next_layer_size:
                output_layer_number = layer_index+1; break

        layer_number = len(self.layers)
        if output_layer_number == -1:
            output_layer_number = layer_number

        fout = smart_open(file_path, 'wb')
        for i in xrange(output_layer_number):
            # decide the dropout factor for this layer
            dropout_factor = 0.0
            if i == 0:
                dropout_factor = self.input_dropout_factor
            if i > 0 and len(self.dropout_factor) > 0:
                dropout_factor = self.dropout_factor[i-1]

            activation_text = '<' + self.cfg.activation_text + '>'
            if i == (layer_number-1) and with_softmax:   # we assume that the last layer is a softmax layer
                activation_text = '<softmax>'
            W_mat = (1.0 - dropout_factor) * self.layers[i].W.get_value()
            b_vec = self.layers[i].b.get_value()
            input_size, output_size = W_mat.shape
            W_layer = []; b_layer = ''
            for rowX in xrange(output_size):
                W_layer.append('')

            for x in xrange(input_size):
                for t in xrange(output_size):
                    W_layer[t] = W_layer[t] + str(W_mat[x][t]) + ' '

            for x in xrange(output_size):
                b_layer = b_layer + str(b_vec[x]) + ' '

            fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
            fout.write('[' + '\n')
            for x in xrange(output_size):
                fout.write(W_layer[x].strip() + '\n')
            fout.write(']' + '\n')
            fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
            if activation_text == '<maxout>':
                fout.write(activation_text + ' ' + str(output_size/self.pool_size) + ' ' + str(output_size) + '\n')
            else:
                fout.write(activation_text + ' ' + str(output_size) + ' ' + str(output_size) + '\n')
        fout.close()
