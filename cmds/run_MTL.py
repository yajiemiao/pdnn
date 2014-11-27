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

from models.dnn import DNN
from models.dropout_nnet import DNN_Dropout

from io_func.model_io import _nnet2file, _cfg2file, _file2nnet, log
from utils.utils import parse_arguments, parse_data_spec_mtl, parse_nnet_spec_mtl
from utils.learn_rates import _lrate2file, _file2lrate

from utils.network_config import NetworkConfig 
from learning.sgd import validate_by_minibatch

# Implements Multi-Task Learning (MTL) in which several tasks share some lower hidden
# layers (shared representation learning). Each task has its specific higher layers (in 
# the simplest case, a task-specific softmax layer). References include:

# J. Huang, J. Li, D. Yu, L. Deng, and Y. Gong. Cross-language knowledge transfer using
# multilingual deep neural network with shared hidden layers. ICASSP 2013.

# Y. Miao, and F. Metze. Improving language-universal feature extraction with deep maxout
# and convolutional neural networks. Interspeech 2014.
 

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements) 

    required_arguments = ['train_data', 'valid_data', 'task_number', 'shared_nnet_spec', 'indiv_nnet_spec', 'wdir']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    train_data_spec = arguments['train_data']; valid_data_spec = arguments['valid_data']
    task_number = int(arguments['task_number'])
    shared_spec = arguments['shared_nnet_spec']; indiv_spec = arguments['indiv_nnet_spec']
    wdir = arguments['wdir']

    # various lists used in MTL
    config_array = [] 
    train_fn_array = []; valid_fn_array = []
    dnn_array = []
    
    # parse data specification
    train_data_spec_array = parse_data_spec_mtl(train_data_spec)
    valid_data_spec_array = parse_data_spec_mtl(valid_data_spec)
    if len(train_data_spec_array) != task_number or len(valid_data_spec_array) != task_number:
        print "Error: #datasets in data specification doesn't match #tasks"; exit(1)
    # split shared_spec ans indiv_spec into individual task's networks
    nnet_spec_array, shared_layers_num = parse_nnet_spec_mtl(shared_spec, indiv_spec)   
    if len(nnet_spec_array) != task_number:
        print "Error: #networks specified by --indiv-spec doesn't match #tasks"; exit(1)
    # parse network configuration from arguments, and initialize data reading
    for n in xrange(task_number):
        network_config = NetworkConfig()
        network_config.parse_config_dnn(arguments, nnet_spec_array[n])
        network_config.init_data_reading(train_data_spec_array[n], valid_data_spec_array[n]) 
        config_array.append(network_config) 

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    resume_training = False; resume_tasks = []  # if we are resuming training, then MLT only operates on the terminated tasks
    for n in xrange(task_number):
        log('> ... building the model for task %d' % (n))
        cfg = config_array[n]
        # set up the model
        dnn_shared = None; shared_layers = []
        if n > 0:
            dnn_shared = dnn_array[0]; shared_layers = [m for m in xrange(shared_layers_num)]
        if cfg.do_dropout:
            dnn = DNN_Dropout(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg,
                              dnn_shared = dnn_shared, shared_layers = shared_layers)
        else:
            dnn = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg,
                      dnn_shared = dnn_shared, shared_layers = shared_layers)

        # get the training, validation and testing function for the model
        log('> ... getting the finetuning functions for task %d' % (n))
        train_fn, valid_fn = dnn.build_finetune_functions((cfg.train_x, cfg.train_y), (cfg.valid_x, cfg.valid_y), batch_size=cfg.batch_size)
        # add dnn and the functions to the list   
        dnn_array.append(dnn)
        train_fn_array.append(train_fn); valid_fn_array.append(valid_fn)
        # check the working dir to decide whether it's resuming training; if yes, load the tmp network files for initialization
        if os.path.exists(wdir + '/nnet.tmp.task' + str(n)) and os.path.exists(wdir + '/training_state.tmp.task' + str(n)):
            resume_training = True; resume_tasks.append(n)
            cfg.lrate = _file2lrate(wdir + '/training_state.tmp.task' + str(n))
            log('> ... found nnet.tmp.task%d and training_state.tmp.task%d, now resume task%d training from epoch %d' % (n, n, n, cfg.lrate.epoch))
            _file2nnet(dnn.layers, filename = wdir + '/nnet.tmp.task' + str(n))

    # pre-training works only if we are NOT resuming training
    # we assume that we only pre-train the shared layers; thus we only use dnn_array[0] to load the parameters
    # because the parameters are shared across the tasks
    ptr_layer_number = 0; ptr_file = ''
    if arguments.has_key('ptr_file') and arguments.has_key('ptr_layer_number'):
        ptr_file = arguments['ptr_file']; ptr_layer_number = int(arguments['ptr_layer_number'])
    if (ptr_layer_number > 0) and (resume_training is False):
        _file2nnet(dnn_array[0].layers, set_layer_num = ptr_layer_number, filename = ptr_file)

    log('> ... finetuning the model')
    train_error_array = [[] for n in xrange(task_number)]
    active_tasks = [n for n in xrange(task_number)]  # the tasks with 0 learning rate are not considered
    if resume_training:
        active_tasks = resume_tasks

    while len(active_tasks) != 0:  # still have tasks which have non-zero learning rate
        # record the mini-batch numbers of the read-in data chunk, on each of the active tasks
        batch_numbers_per_chunk = [0 for n in xrange(task_number)]
        for n in active_tasks:
            config_array[n].train_sets.load_next_partition(config_array[n].train_xy)
            batch_numbers_per_chunk[n] = config_array[n].train_sets.cur_frame_num / config_array[n].batch_size
        # although we set one single trunk size, the actual size of data chunks we read in may differ
        # across the tasks. this is because we may reach the end of the data file. thus, we loop over
        # the max number of mini-batches, but do the checking on each individual task 
        for batch_index in xrange(max(batch_numbers_per_chunk)):  # loop over mini-batches
            for n in active_tasks:
                if batch_index < batch_numbers_per_chunk[n]:
                    train_error_array[n].append(train_fn_array[n](index=batch_index, learning_rate = config_array[n].lrate.get_rate(), momentum = config_array[n].momentum))

        # now check whether we finish one epoch on any of the tasks
        active_tasks_new = active_tasks
        for n in active_tasks:
            cfg = config_array[n]
            if cfg.train_sets.is_finish():  # if true, we reach the end of one epoch of task #n
                # reset data reading to the start of next epoch, and output the training error
                cfg.train_sets.initialize_read()
                log('> task %d, epoch %d, training error %f ' % (n, cfg.lrate.epoch, 100*numpy.mean(train_error_array[n])) + '(%)')
                train_error_array[n] = []
                # perform validation, output valid error rate, and adjust learning rate based on the learning rate
                valid_error = validate_by_minibatch(valid_fn_array[n], cfg)
                log('> task %d, epoch %d, lrate %f, validation error %f ' % (n, cfg.lrate.epoch, cfg.lrate.get_rate(), 100*numpy.mean(valid_error)) + '(%)')
                cfg.lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))
                # output nnet parameters and lrate, for training resume
                _nnet2file(dnn_array[n].layers, filename=wdir + '/nnet.tmp.task' + str(n))
                _lrate2file(cfg.lrate, wdir + '/training_state.tmp.task' + str(n))
                # if the lrate of a task decays to 0, training on this task terminates; it will be excluded from future training
                if cfg.lrate.get_rate() == 0:
                    active_tasks_new.remove(n)
                    # save the model and network configuration
                    if cfg.param_output_file != '':
                        _nnet2file(dnn_array[n].layers, filename=cfg.param_output_file + '.task' + str(n),
                                   input_factor = cfg.input_dropout_factor, factor = cfg.dropout_factor)
                        log('> ... the final PDNN model parameter is ' + cfg.param_output_file + '.task' + str(n))
                    if cfg.cfg_output_file != '':
                        _cfg2file(dnn_array[n].cfg, filename=cfg.cfg_output_file + '.task' + str(n))
                        log('> ... the final PDNN model config is ' + cfg.cfg_output_file + '.task' + str(n))
                    # output the model into Kaldi-compatible format
                    if cfg.kaldi_output_file != '':
                        dnn_array[n].write_model_to_kaldi(cfg.kaldi_output_file + '.task' + str(n))
                        log('> ... the final Kaldi model is ' + cfg.kaldi_output_file + '.task' + str(n))

                    # remove the tmp files (which have been generated from resuming training)
                    os.remove(wdir + '/nnet.tmp.task' + str(n)); os.remove(wdir + '/training_state.tmp.task' + str(n))
        active_tasks = active_tasks_new

