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

# Various functions to convert model into Janus formats

import sys, os, struct
import sys

from StringIO import StringIO
import json

# write a matrix into the matlab format
def write_mat_matlab(input_size, output_size, name, mat, fout = sys.stdout):
    fout.write(struct.pack('5I%dsb' % len(name), 10, output_size, input_size, 0, len(name)+1, name, 0))
    for m in xrange(input_size):
        for n in xrange(output_size):
            fout.write(struct.pack('f', mat[n][m]))

# write a vector into the matlab format
def write_vec_matlab(output_size, name, vec, fout = sys.stdout):
    fout.write(struct.pack('5I%dsb' % len(name), 10, 1, output_size, 0, len(name)+1, name, 0))
    for m in xrange(output_size):
        fout.write(struct.pack('f', vec[m]))

def _nnet2janus(nnet_topo, set_layer_num = -1, filein='nnet.in', fileout='nnet.out', activation='sigmoid', withfinal=True):

    layers = nnet_topo.split(':')
    if set_layer_num == -1:
        layer_num = len(layers) - 1
    else:
        layer_num = set_layer_num + 1
    nnet_dict = {}
    with open(filein, 'rb') as fp:
        nnet_dict = json.load(fp)
    fout = open(fileout, 'wb')
    lnum = 1
    for i in xrange(layer_num - 1):
        input_size = int(layers[i])
        output_size = int(layers[i + 1])
        W_layer = [[0 for x in xrange(input_size)] for x in xrange(output_size)]
        b_layer = [0 for x in xrange(output_size)]

        dict_key = str(i) + ' ' + activation + ' W'
        matrix_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(input_size):
            elements = matrix_rows[x].split(' ')
            for t in xrange(output_size):
                W_layer[t][x] = float(elements[t])

        dict_key = str(i) + ' ' + activation + ' b'
        vector_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(output_size):
            b_layer[x] = float(vector_rows[x])

        write_mat_matlab(name='weights%d%d' % (lnum, lnum+1), input_size=input_size,
                  output_size=output_size, mat=W_layer, fout = fout)
        write_vec_matlab(name='bias%d' % (lnum+1), output_size=output_size, vec=b_layer, fout = fout)

        lnum = lnum + 1

    if withfinal:
        input_size = int(layers[-2])
        output_size = int(layers[-1])
        W_layer = [[0 for x in xrange(input_size)] for x in xrange(output_size)]
        b_layer = [0 for x in xrange(output_size)]

        dict_key = 'logreg W'
        matrix_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(input_size):
            elements = matrix_rows[x].split(' ')
            for t in xrange(output_size):
                W_layer[t][x] = float(elements[t])


        dict_key = 'logreg b'
        vector_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(output_size):
            b_layer[x] = float(vector_rows[x])

        write_mat_matlab(name='weights%d%d' % (lnum, lnum+1), input_size=input_size,
                  output_size=output_size, mat=W_layer, fout = fout)

        write_vec_matlab(name='bias%d' % (lnum+1), output_size=output_size, vec=b_layer, fout = fout)

    fout.close()

def _nnet2janus_maxout(nnet_topo, set_layer_num = -1, pool_size = 1, filein='nnet.in', fileout='nnet.out', activation='sigmoid', withfinal=True):

    layers = nnet_topo.split(':')
    if set_layer_num == -1:
        layer_num = len(layers) - 1
    else:
        layer_num = set_layer_num + 1
    nnet_dict = {}
    with open(filein, 'rb') as fp:
        nnet_dict = json.load(fp)
    fout = open(fileout, 'wb')
    lnum = 1
    for i in xrange(layer_num - 1):
        input_size = int(layers[i])
        output_size = int(layers[i + 1]) * pool_size
        W_layer = [[0 for x in xrange(input_size)] for x in xrange(output_size)]
        b_layer = [0 for x in xrange(output_size)]

        dict_key = str(i) + ' ' + activation + ' W'
        matrix_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(input_size):
            elements = matrix_rows[x].split(' ')
            for t in xrange(output_size):
                W_layer[t][x] = float(elements[t])

        dict_key = str(i) + ' ' + activation + ' b'
        vector_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(output_size):
            b_layer[x] = float(vector_rows[x])

        write_mat_matlab(name='weights%d%d' % (lnum, lnum+1), input_size=input_size,
                  output_size=output_size, mat = W_layer, fout = fout)
        write_vec_matlab(name='bias%d' % (lnum+1), output_size=output_size, vec=b_layer, fout = fout)

        lnum = lnum + 1

    if withfinal:
        input_size = int(layers[-2])
        output_size = int(layers[-1])
        W_layer = [[0 for x in xrange(input_size)] for x in xrange(output_size)]
        b_layer = [0 for x in xrange(output_size)]

        dict_key = 'logreg W'
        matrix_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(input_size):
            elements = matrix_rows[x].split(' ')
            for t in xrange(output_size):
                W_layer[t][x] = float(elements[t])


        dict_key = 'logreg b'
        vector_rows = nnet_dict[dict_key].split('\n')
        for x in xrange(output_size):
            b_layer[x] = float(vector_rows[x])

        write_mat_matlab(name='weights%d%d' % (lnum, lnum+1), input_size=input_size,
                  output_size=output_size, mat = W_layer, fout = fout)

        write_vec_matlab(name='bias%d' % (lnum+1), output_size=output_size, vec=b_layer, fout = fout)

    fout.close()


