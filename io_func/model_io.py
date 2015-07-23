# Copyright 2013    Yajie Miao    Carnegie Mellon University
#           2015    Yun Wang      Carnegie Mellon University

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

# Various functions to write models from nets to files, and to read models from
# files to nets

import numpy as np
import os
import sys
import cPickle

from StringIO import StringIO
import json

import theano
import theano.tensor as T

from datetime import datetime

from io_func import smart_open

# print log to standard output
def log(string):
    sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')

# convert an array to a string
def array_2_string(array):
    str_out = StringIO()
    np.savetxt(str_out, array)
    return str_out.getvalue()

# convert a string to an array
#def string_2_array(string):
#    str_in = StringIO(string)
#    return np.loadtxt(str_in)

def string_2_array(string):
    str_in = StringIO(string)
    array_tmp = np.loadtxt(str_in)
    if len(array_tmp.shape) == 0:
        return np.array([array_tmp])
    return array_tmp

def _nnet2file(layers, set_layer_num = -1, filename='nnet.out', start_layer = 0, input_factor = 0.0, factor=[]):
    n_layers = len(layers)
    nnet_dict = {}
    if set_layer_num == -1:
       set_layer_num = n_layers

    for i in range(start_layer, set_layer_num):
       layer = layers[i]
       dict_a = 'W' + str(i)
       dropout_factor = 0.0
       if i == 0:
           dropout_factor = input_factor
       if i > 0 and len(factor) > 0:
           dropout_factor = factor[i-1]

       if layer.type == 'fc':
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W.get_value())
       elif layer.type == 'conv':
           filter_shape = layer.filter_shape
           for next_X in xrange(filter_shape[0]):
               for this_X in xrange(filter_shape[1]):
                   new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                   nnet_dict[new_dict_a] = array_2_string((1.0-dropout_factor) * (layer.W.get_value())[next_X, this_X])

       dict_a = 'b' + str(i)
       nnet_dict[dict_a] = array_2_string(layer.b.get_value())

    with smart_open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()


# save the config classes; since we are using pickle to serialize the whole class, it's better to set the
# data reading and learning rate interfaces to None.
def _cfg2file(cfg, filename='cfg.out'):
    cfg.lrate = None
    cfg.train_sets = None; cfg.train_xy = None; cfg.train_x = None; cfg.train_y = None
    cfg.valid_sets = None; cfg.valid_xy = None; cfg.valid_x = None; cfg.valid_y = None
    cfg.activation = None  # saving the rectifier function causes errors; thus we don't save the activation function
                           # the activation function is initialized from the activation text ("sigmoid") when the network
                           # configuration is loaded
    with smart_open(filename, "wb") as output:
        cPickle.dump(cfg, output, cPickle.HIGHEST_PROTOCOL)

def _file2nnet(layers, set_layer_num = -1, filename='nnet.in',  factor=1.0):
    n_layers = len(layers)
    nnet_dict = {}
    if set_layer_num == -1:
        set_layer_num = n_layers

    with smart_open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)
    for i in xrange(set_layer_num):
        dict_a = 'W' + str(i)
        layer = layers[i]
        if layer.type == 'fc':
            mat_shape = layer.W.get_value().shape
            layer.W.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX).reshape(mat_shape))
        elif layer.type == 'conv':
            filter_shape = layer.filter_shape
            W_array = layer.W.get_value()
            for next_X in xrange(filter_shape[0]):
                for this_X in xrange(filter_shape[1]):
                    new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                    mat_shape = W_array[next_X, this_X, :, :].shape
                    W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[new_dict_a]), dtype=theano.config.floatX).reshape(mat_shape)
            layer.W.set_value(W_array)
        dict_a = 'b' + str(i)
        layer.b.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))

def _cnn2file(conv_layers, filename='nnet.out', input_factor = 1.0, factor=[]):
    n_layers = len(conv_layers)
    nnet_dict = {}
    for i in xrange(n_layers):
       conv_layer = conv_layers[i]
       filter_shape = conv_layer.filter_shape

       dropout_factor = 0.0
       if i == 0:
           dropout_factor = input_factor
       if i > 0 and len(factor) > 0:
           dropout_factor = factor[i-1]

       for next_X in xrange(filter_shape[0]):
           for this_X in xrange(filter_shape[1]):
               dict_a = 'W' + str(i) + ' ' + str(next_X) + ' ' + str(this_X)
               nnet_dict[dict_a] = array_2_string(dropout_factor * (conv_layer.W.get_value())[next_X, this_X])

       dict_a = 'b' + str(i)
       nnet_dict[dict_a] = array_2_string(conv_layer.b.get_value())

    with smart_open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()

def _file2cnn(conv_layers, filename='nnet.in', factor=1.0):
    n_layers = len(conv_layers)
    nnet_dict = {}

    with smart_open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)
    for i in xrange(n_layers):
        conv_layer = conv_layers[i]
        filter_shape = conv_layer.filter_shape
        W_array = conv_layer.W.get_value()

        for next_X in xrange(filter_shape[0]):
            for this_X in xrange(filter_shape[1]):
                dict_a = 'W' + str(i) + ' ' + str(next_X) + ' ' + str(this_X)
                W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[dict_a]))

        conv_layer.W.set_value(W_array)

        dict_a = 'b' + str(i)
        conv_layer.b.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
