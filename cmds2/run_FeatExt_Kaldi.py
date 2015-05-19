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

from models.cnn import CNN
from models.dnn import DNN
from utils.utils import parse_arguments

if __name__ == '__main__':


    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['in_scp_file', 'out_ark_file', 'nnet_param', 'nnet_cfg', 'layer_index']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    in_scp_file = arguments['in_scp_file']
    out_ark_file = arguments['out_ark_file']
    nnet_param = arguments['nnet_param']
    nnet_cfg = arguments['nnet_cfg']
    layer_index = int(arguments['layer_index'])

    # load network configuration
    cfg = cPickle.load(smart_open(nnet_cfg,'r'))
    cfg.init_activation()

    # set up the model with model config
    log('> ... setting up the model and loading parameters')
    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    cfg = cPickle.load(smart_open(nnet_cfg,'r'))
    model = None
    if cfg.model_type == 'DNN':
        model = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)
    elif cfg.model_type == 'CNN':
        model = CNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg, testing = True)

    # load model parameters
    _file2nnet(model.layers, filename = nnet_param)

    # get the function for feature extraction
    log('> ... getting the feat-extraction function')
    extract_func = model.build_extract_feat_function(layer_index)

    kaldiread = KaldiReadIn(in_scp_file)
    kaldiwrite = KaldiWriteOut(out_ark_file)
    log('> ... processing the data')
    utt_number = 0
    while True:
        uttid, in_matrix = kaldiread.read_next_utt()
        if uttid == '':
            break
#        in_matrix = numpy.reshape(in_matrix, (in_matrix.shape[0],) + input_shape_1)
        out_matrix = extract_func(in_matrix)
        kaldiwrite.write_kaldi_mat(uttid, out_matrix)
        utt_number += 1
        if utt_number % 100 == 0:
            log('> ... processed %d utterances' % (utt_number))

    kaldiwrite.close()

    log('> ... the saved features are %s' % (out_ark_file))
