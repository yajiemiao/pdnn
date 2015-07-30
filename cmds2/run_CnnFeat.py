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
import json

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from io_func import smart_open
from io_func.model_io import _file2nnet, log
from io_func.kaldi_feat import KaldiReadIn, KaldiWriteOut

from models.cnn import CNN_Forward
from utils.utils import parse_arguments, string2bool

if __name__ == '__main__':


    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['in_scp_file', 'out_ark_file', 'cnn_param_file', 'cnn_cfg_file']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    in_scp_file = arguments['in_scp_file']
    out_ark_file = arguments['out_ark_file']
    cnn_param_file = arguments['cnn_param_file']
    cnn_cfg_file = arguments['cnn_cfg_file']
    # network structure
    cfg = cPickle.load(smart_open(cnn_cfg_file,'r'))

    conv_configs = cfg.conv_layer_configs
    conv_layer_number = len(conv_configs)
    for i in xrange(conv_layer_number):
        conv_configs[i]['activation'] = cfg.conv_activation

    # whether to use the fast mode
    use_fast = cfg.use_fast
    if arguments.has_key('use_fast'):
        use_fast = string2bool(arguments['use_fast'])

    kaldiread = KaldiReadIn(in_scp_file)
    kaldiwrite = KaldiWriteOut(out_ark_file)


    log('> ... setting up the CNN convolution layers')
    input_shape_train = conv_configs[0]['input_shape']
    input_shape_1 = (input_shape_train[1], input_shape_train[2], input_shape_train[3])

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    cnn = CNN_Forward(numpy_rng = rng, theano_rng=theano_rng, conv_layer_configs = conv_configs, use_fast = use_fast)
    _file2nnet(cnn.conv_layers, set_layer_num = len(conv_configs), filename=cnn_param_file)
    out_function = cnn.build_out_function()

    log('> ... processing the data')

    while True:
        uttid, in_matrix = kaldiread.read_next_utt()
        if uttid == '':
            break
        in_matrix = numpy.reshape(in_matrix, (in_matrix.shape[0],) + input_shape_1)
        out_matrix = out_function(in_matrix)
        kaldiwrite.write_kaldi_mat(uttid, out_matrix)

    kaldiwrite.close()

    log('> ... the saved features are %s' % (out_ark_file))
