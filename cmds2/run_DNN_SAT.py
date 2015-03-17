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

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from models.dnn_sat import DNN_SAT

from io_func.model_io import _nnet2file, _file2nnet, log
from utils.utils import parse_arguments

from learning.sgd import train_sgd_verbose, validate_by_minibatch_verbose 
from utils.network_config import NetworkConfig

# Implements the Speaker Adaptive Training of DNNs proposed in the following papers:

# [1] Yajie Miao, Hao Zhang, Florian Metze. "Towards Speaker Adaptive Training of Deep
#  Neural Network Acoustic Models". Interspeech 2014.

# [2] Yajie Miao, Lu Jiang, Hao Zhang, Florian Metze. "Improvements to Speaker Adaptive
# Training of Deep Neural Networks". SLT 2014.

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['train_data', 'valid_data', 'si_nnet_spec', 'wdir', 'adapt_nnet_spec', 'init_model']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    train_data_spec = arguments['train_data']; valid_data_spec = arguments['valid_data']
    si_nnet_spec = arguments['si_nnet_spec']
    adapt_nnet_spec = arguments['adapt_nnet_spec'];
    wdir = arguments['wdir']
    init_model_file = arguments['init_model']

    # parse network configuration from arguments, and initialize data reading
    cfg_si = NetworkConfig()
    cfg_si.parse_config_dnn(arguments, si_nnet_spec)
    cfg_si.init_data_reading(train_data_spec, valid_data_spec)   
    # parse the structure of the i-vector network 
    cfg_adapt = NetworkConfig()
#    net_split = adapt_nnet_spec.split(':')
#    adapt_nnet_spec = ''
#    for n in xrange(len(net_split) - 1):
#        adapt_nnet_spec += net_split[n] + ':'
#    cfg_adapt.parse_config_dnn(arguments, adapt_nnet_spec + '0')
    cfg_adapt.parse_config_dnn(arguments, adapt_nnet_spec + ':0')

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    log('> ... initializing the model')
    # setup up the model 
    dnn = DNN_SAT(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg_si = cfg_si, cfg_adapt = cfg_adapt)
    # read the initial DNN  (the SI DNN which has been well trained)
    _file2nnet(dnn.dnn_si.layers, filename = init_model_file)

    # get the training and  validation functions for adaptation network training
    dnn.params = dnn.dnn_adapt.params  # only update the parameters of the i-vector nnet
    dnn.delta_params = dnn.dnn_adapt.delta_params
    log('> ... getting the finetuning functions for iVecNN')
    train_fn, valid_fn = dnn.build_finetune_functions(
                (cfg_si.train_x, cfg_si.train_y), (cfg_si.valid_x, cfg_si.valid_y),
                batch_size = cfg_adapt.batch_size)

    log('> ... learning the adaptation network')
    cfg = cfg_adapt
    while (cfg.lrate.get_rate() != 0):
        # one epoch of sgd training
        train_error = train_sgd_verbose(train_fn, cfg_si.train_sets, cfg_si.train_xy,
                                        cfg.batch_size, cfg.lrate.get_rate(), cfg.momentum)
        log('> epoch %d, training error %f ' % (cfg.lrate.epoch, 100*numpy.mean(train_error)) + '(%)')
        # validation
        valid_error = validate_by_minibatch_verbose(valid_fn, cfg_si.valid_sets, cfg_si.valid_xy, cfg.batch_size)
        log('> epoch %d, lrate %f, validation error %f ' % (cfg.lrate.epoch, cfg.lrate.get_rate(), 100*numpy.mean(valid_error)) + '(%)')
        cfg.lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))

    # get the training and validation function for SI DNN re-updating
    dnn.params = dnn.dnn_si.params  # now only update the parameters of the SI DNN
    dnn.delta_params = dnn.dnn_si.delta_params
    log('> ... getting the finetuning functions for DNN')
    train_fn, valid_fn = dnn.build_finetune_functions(
                (cfg_si.train_x, cfg_si.train_y), (cfg_si.valid_x, cfg_si.valid_y),
                batch_size = cfg_si.batch_size)

    log('> ... learning the DNN model in the new feature space')
    cfg = cfg_si
    while (cfg.lrate.get_rate() != 0):
        # one epoch of sgd training
        train_error = train_sgd_verbose(train_fn, cfg_si.train_sets, cfg_si.train_xy,
                                        cfg.batch_size, cfg.lrate.get_rate(), cfg.momentum)
        log('> epoch %d, training error %f ' % (cfg.lrate.epoch, 100*numpy.mean(train_error)) + '(%)')
        # validation
        valid_error = validate_by_minibatch_verbose(valid_fn, cfg_si.valid_sets, cfg_si.valid_xy, cfg.batch_size)
        log('> epoch %d, lrate %f, validation error %f ' % (cfg.lrate.epoch, cfg.lrate.get_rate(), 100*numpy.mean(valid_error)) + '(%)')
        cfg.lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))

    # save the model and network configuration
    if cfg.param_output_file != '':
        _nnet2file(dnn.dnn_adapt.layers, filename = cfg.param_output_file + '.adapt',
                   input_factor = cfg_adapt.input_dropout_factor, factor = cfg_adapt.dropout_factor)
        _nnet2file(dnn.dnn_si.layers, filename = cfg.param_output_file + '.si',
                   input_factor = cfg_si.input_dropout_factor, factor = cfg_si.dropout_factor)
        log('> ... the final PDNN model parameter is ' + cfg.param_output_file + ' (.si, .adapt)')
    if cfg.cfg_output_file != '':
        _cfg2file(cfg_adapt, filename=cfg.cfg_output_file + '.adapt')
        _cfg2file(cfg_si, filename=cfg.cfg_output_file + '.si')
        log('> ... the final PDNN model config is ' + cfg.cfg_output_file + ' (.si, .adapt)')

    # output the model into Kaldi-compatible format
    if cfg.kaldi_output_file != '':
        dnn.dnn_si.write_model_to_kaldi(cfg.kaldi_output_file + '.si')
        dnn.dnn_adapt.write_model_to_kaldi(cfg.kaldi_output_file + '.adapt', with_softmax = False)
        log('> ... the final Kaldi model is ' + cfg.kaldi_output_file + ' (.si, .adapt)')
