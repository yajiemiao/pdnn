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

from models.srbm import SRBM
from models.dnn import DNN
from utils.network_config import NetworkConfig
from utils.rbm_config import RBMConfig

from io_func.model_io import _nnet2file, _cfg2file,  _file2nnet, log
from utils.utils import parse_arguments, save_two_integers, read_two_integers

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['train_data', 'nnet_spec', 'wdir']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    train_data_spec = arguments['train_data']
    nnet_spec = arguments['nnet_spec']
    wdir = arguments['wdir']

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    log('> ... initializing the model')

    # parse network configuration from arguments, and initialize data reading
    cfg = RBMConfig()
    cfg.parse_config_common(arguments)
    cfg.init_data_reading(train_data_spec)

    # we also need to set up a DNN model, whose parameters are shared with RBM, for 2 reasons:
    # first, we can use DNN's model reading and writing functions, instead of designing these functions for RBM specifically
    # second, DNN generates 
    cfg_dnn = NetworkConfig()
    cfg_dnn.n_ins = cfg.n_ins; cfg_dnn.hidden_layers_sizes = cfg.hidden_layers_sizes; cfg_dnn.n_outs = cfg.n_outs
    dnn = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg_dnn)

    # now set up the RBM model with dnn as an argument
    srbm = SRBM(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg, dnn = dnn)
    # get the pre-training function
    log('> ... getting the pre-training functions')
    pretraining_fns = srbm.pretraining_functions(train_set_x=cfg.train_x, batch_size=cfg.batch_size,
                                                 k = 1, weight_cost = 0.0002)
 
    start_layer_index = 0
    start_epoch_index = 0
    if os.path.exists(wdir + '/nnet.tmp') and os.path.exists(wdir + '/training_state.tmp'):
        start_layer_index, start_epoch_index = read_two_integers(wdir + '/training_state.tmp')
        log('> ... found nnet.tmp and training_state.tmp, now resume training from layer #' + str(start_layer_index) + ' epoch #' + str(start_epoch_index))
        _file2nnet(dnn.layers, filename = wdir + '/nnet.tmp')

    log('> ... training the model')
    log('> r_cost = reconstruction cost, fe_cost = (approximate) value of free energy function') 
    # pre-train layer-wise
    for i in xrange(start_layer_index, cfg.ptr_layer_number):
        if (srbm.rbm_layers[i].is_gbrbm()):
            pretrain_lr = cfg.gbrbm_learning_rate
        else:
            pretrain_lr = cfg.learning_rate
        # go through pretraining epochs
        momentum = cfg.initial_momentum
        for epoch in range(start_epoch_index, cfg.epochs):
            # go through the training set
            if (epoch == cfg.initial_momentum_epoch):
                momentum = cfg.final_momentum

            r_c, fe_c = [], []  # keep record of reconstruction and free-energy cost
            while (not cfg.train_sets.is_finish()):
                cfg.train_sets.load_next_partition(cfg.train_xy)
                for batch_index in xrange(cfg.train_sets.cur_frame_num / cfg.batch_size):  # loop over mini-batches
                    [reconstruction_cost, free_energy_cost] = pretraining_fns[i](index=batch_index, lr=pretrain_lr,
                                                                                 momentum=momentum)
                    r_c.append(reconstruction_cost)
                    fe_c.append(free_energy_cost)
            cfg.train_sets.initialize_read()
            log('> pre-training layer %i, epoch %d, r_cost %f, fe_cost %f' % (i, epoch, numpy.mean(r_c), numpy.mean(fe_c)))
            # output nnet parameters and training state, for training resume
            _nnet2file(dnn.layers, filename=wdir + '/nnet.tmp')
            save_two_integers((i, epoch+1), wdir + '/training_state.tmp')
        start_epoch_index = 0
        save_two_integers((i+1, 0), wdir + '/training_state.tmp')

    # save the pretrained nnet to file
    # save the model and network configuration
    if cfg.param_output_file != '':
        _nnet2file(dnn.layers, filename=cfg.param_output_file)
        log('> ... the final PDNN model parameter is ' + cfg.param_output_file)
    if cfg.cfg_output_file != '':
        _cfg2file(dnn.cfg, filename=cfg.cfg_output_file)
        log('> ... the final PDNN model config is ' + cfg.cfg_output_file)

    # output the model into Kaldi-compatible format
    if cfg.kaldi_output_file != '':
        dnn.write_model_to_kaldi(cfg.kaldi_output_file)
        log('> ... the final Kaldi model is ' + cfg.kaldi_output_file) 

    # finally remove the training-resuming files
    os.remove(wdir + '/nnet.tmp')
    os.remove(wdir + '/training_state.tmp')
