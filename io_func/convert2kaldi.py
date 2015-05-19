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

import numpy as np
import os
import sys

from StringIO import StringIO
import json

from io_func import smart_open

# Various functions to convert models into Kaldi formats
def _nnet2kaldi(nnet_spec, set_layer_num = -1, filein='nnet.in',
               fileout='nnet.out', activation='sigmoid', withfinal=True):

    elements = nnet_spec.split(":")
    layers = []
    for x in elements:
        layers.append(int(x))

    if set_layer_num == -1:
        layer_num = len(layers) - 1
    else:
        layer_num = set_layer_num + 1
    nnet_dict = {}
    with smart_open(filein, 'rb') as fp:
        nnet_dict = json.load(fp)
    fout = smart_open(fileout, 'wb')
    for i in xrange(layer_num - 1):
        input_size = int(layers[i])
        output_size = int(layers[i + 1])
        W_layer = []
        b_layer = ''
        for rowX in xrange(output_size):
            W_layer.append('')

        dict_key = str(i) + ' ' + activation + ' W'
        matrix_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(input_size):
            elements = matrix_rows[x].split(' ')
            for t in xrange(output_size):
                W_layer[t] = W_layer[t] + str(float(elements[t])) + ' '

        dict_key = str(i) + ' ' + activation + ' b'
        vector_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(output_size):
            b_layer = b_layer + str(float(vector_rows[x])) + ' '

        fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
        fout.write('[' + '\n')
        for x in xrange(output_size):
            fout.write(W_layer[x].strip() + '\n')
        fout.write(']' + '\n')
        fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
        fout.write('<sigmoid> ' + str(output_size) + ' ' + str(output_size) + '\n')


    if withfinal:
        input_size = int(layers[-2])
        output_size = int(layers[-1])
        W_layer = []
        b_layer = ''
        for rowX in xrange(output_size):
            W_layer.append('')

        dict_key = 'logreg W'
        matrix_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(input_size):
            elements = matrix_rows[x].split(' ')
            for t in xrange(output_size):
                W_layer[t] = W_layer[t] + str(float(elements[t])) + ' '


        dict_key = 'logreg b'
        vector_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(output_size):
            b_layer = b_layer + str(float(vector_rows[x])) + ' '

        fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
        fout.write('[' + '\n')
        for x in xrange(output_size):
            fout.write(W_layer[x].strip() + '\n')
        fout.write(']' + '\n')
        fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
        fout.write('<softmax> ' + str(output_size) + ' ' + str(output_size) + '\n')

    fout.close();

def _nnet2kaldi_maxout(nnet_spec, pool_size = 1, set_layer_num = -1,
                      filein='nnet.in', fileout='nnet.out', activation='sigmoid', withfinal=True):

    elements = nnet_spec.split(':')
    layers = []
    for x in elements:
        layers.append(int(x))
    if set_layer_num == -1:
        layer_num = len(layers) - 1
    else:
        layer_num = set_layer_num + 1
    nnet_dict = {}
    with smart_open(filein, 'rb') as fp:
        nnet_dict = json.load(fp)
    fout = smart_open(fileout, 'wb')
    for i in xrange(layer_num - 1):
        input_size = int(layers[i])
        output_size = int(layers[i + 1]) * pool_size
        W_layer = []
        b_layer = ''
        for rowX in xrange(output_size):
            W_layer.append('')

        dict_key = str(i) + ' ' + activation + ' W'
        matrix_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(input_size):
            elements = matrix_rows[x].split(' ')
            for t in xrange(output_size):
                W_layer[t] = W_layer[t] + str(float(elements[t])) + ' '


        dict_key = str(i) + ' ' + activation + ' b'
        vector_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(output_size):
            b_layer = b_layer + str(float(vector_rows[x])) + ' '

        fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
        fout.write('[' + '\n')
        for x in xrange(output_size):
            fout.write(W_layer[x].strip() + '\n')
        fout.write(']' + '\n')
        fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
        fout.write('<maxout> ' + str(int(layers[i + 1])) + ' ' + str(output_size) + '\n')

    if withfinal:
        input_size = int(layers[-2])
        output_size = int(layers[-1])
        W_layer = []
        b_layer = ''
        for rowX in xrange(output_size):
            W_layer.append('')

        dict_key = 'logreg W'
        matrix_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(input_size):
            elements = matrix_rows[x].split(' ')
            for t in xrange(output_size):
                W_layer[t] = W_layer[t] + str(float(elements[t])) + ' '


        dict_key = 'logreg b'
        vector_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(output_size):
            b_layer = b_layer + str(float(vector_rows[x])) + ' '

        fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
        fout.write('[' + '\n')
        for x in xrange(output_size):
            fout.write(W_layer[x].strip() + '\n')
        fout.write(']' + '\n')
        fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
        fout.write('<softmax> ' + str(output_size) + ' ' + str(output_size) + '\n')

    fout.close();

def _nnet2kaldi_direct(dnn, output_layer_number = -1, fileout='nnet.out'):

    layer_number = len(dnn.sigmoid_layers)
    if output_layer_number == -1:
        output_layer_number = layer_number

    for i in xrange(output_layer_number):
        activation_text = '<' + dnn.cfg.activation_text + '>'
#        activation_text = '<' + activation_text + '>'
        if i == (layer_number-1):   # we assume that the last layer is a softmax layer
            activation_text = '<softmax>'
        W_mat = dnn.sigmoid_layers[i].W.get_value()
        b_vec = dnn.sigmoid_layers[i].b.get_value()
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
        fout.write(activation_text + ' ' + str(output_size) + ' ' + str(output_size) + '\n')

