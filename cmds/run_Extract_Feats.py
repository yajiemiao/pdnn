# Copyright 2014    Yajie Miao    Carnegie Mellon University
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

import cPickle
import gzip
import os
import sys
import time
import math

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from models.dnn import DNN
from models.cnn import CNN

from io_func import smart_open
from io_func.model_io import _file2nnet, log
from utils.utils import parse_arguments, string2bool
from utils.network_config import NetworkConfig

if __name__ == '__main__':

    import sys

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['data', 'nnet_param', 'nnet_cfg', 'output_file', 'layer_index', 'batch_size']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    data_spec = arguments['data']
    nnet_param = arguments['nnet_param']
    nnet_cfg = arguments['nnet_cfg']
    output_file = arguments['output_file']
    layer_index = int(arguments['layer_index'])
    batch_size = float(arguments['batch_size'])
    argmax = arguments.has_key('argmax') and string2bool(arguments['argmax'])

    # load network configuration and set up the model
    log('> ... setting up the model and loading parameters')
    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    cfg = cPickle.load(smart_open(nnet_cfg,'r'))
    cfg.init_activation()
    model = None
    if cfg.model_type == 'DNN':
        model = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)
    elif cfg.model_type == 'CNN':
        model = CNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg, testing = True)

    # load model parameters
    _file2nnet(model.layers, filename = nnet_param)

    # initialize data reading
    cfg.init_data_reading_test(data_spec)

    # get the function for feature extraction
    log('> ... getting the feat-extraction function')
    extract_func = model.build_extract_feat_function(layer_index)

    output_mats = []    # store the features for all the data in memory. TODO: output the features in a streaming mode
    log('> ... generating features from the specified layer')
    while (not cfg.test_sets.is_finish()):  # loop over the data
        cfg.test_sets.load_next_partition(cfg.test_xy)
        batch_num = int(math.ceil(1.0 * cfg.test_sets.cur_frame_num / batch_size))

        for batch_index in xrange(batch_num):  # loop over mini-batches
            start_index = batch_index * batch_size
            end_index = min((batch_index+1) * batch_size, cfg.test_sets.cur_frame_num)  # the residue may be smaller than a mini-batch
            output = extract_func(cfg.test_x.get_value()[start_index:end_index])
            output_mats.append(output)

    output_mat = numpy.concatenate(output_mats)
    if argmax:
        output_mat = output_mat.argmax(axis = 1)

    # output the feature representations using pickle
    f = smart_open(output_file, 'wb')
    cPickle.dump(output_mat, f, cPickle.HIGHEST_PROTOCOL)

    log('> ... the features are stored in ' + output_file)
