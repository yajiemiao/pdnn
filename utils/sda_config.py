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

import theano
import theano.tensor as T
from io_func.data_io import read_data_args, read_dataset
from learn_rates import LearningRateExpDecay
from utils import parse_lrate, parse_activation, parse_conv_spec, activation_to_txt, string2bool


class SdAConfig():

    def __init__(self):

        # parameters related with training 
        self.epochs = 5                  # number of training epochs for each layer
        self.batch_size = 128            # size of mini-batches
        self.corruption_levels=[0.2 for n in xrange(100)]  # denoising factor; we use an array for future extension to layer-specific factor
        self.learning_rates = [.01 for n in xrange(100)]   # learning rate for each layer
        self.momentum = 0                # momentum 

        self.ptr_layer_number = 0        # number of layers to be trained
        self.hidden_activation = T.nnet.sigmoid   # activation function of the hidden layer/output
        self.firstlayer_reconstruct_activation = T.nnet.sigmoid      # the reconstruction activation function for the first layer
                                                                   # if we normaze the input data with mean (maybe also with variance)
                                                                   # normalization, then we need the tanh activation function to reconstruct
                                                                   # the input
        
        # for maxout autoencoder
        self.do_maxout = False           # whether to apply maxout on the hidden layer
        self.pool_size = 1               # pooling size of maxout

        # interfaces for the training data
        self.train_sets = None
        self.train_xy = None
        self.train_x = None
        self.train_y = None

        # interfaces for validation data. we don't do validation for RBM, so these variables will be None 
        # we have these variables because we want to use the _cfg2file function from io_func/model_io.py
        self.valid_sets = None
        self.valid_xy = None
        self.valid_x = None
        self.valid_y = None

        # network structures. same as DNN
        self.n_ins = 0                  # dimension of inputs
        self.hidden_layers_sizes = []   # number of units in the hidden layers
        self.n_outs = 0                 # number of outputs (classes), not used for RBM

        # parse sparsity settings for sparsity autoencoder
        self.sparsity = None
        self.sparsity_weight = None

        # path to save model
        self.cfg_output_file = ''       # where to save this config class
        self.param_output_file = ''     # where to save the network parameters
        self.kaldi_output_file = ''     # where to save the Kaldi-formatted model

    # initialize pfile reading. TODO: inteference *directly* for Kaldi feature and alignment files
    def init_data_reading(self, train_data_spec):
        train_dataset, train_dataset_args = read_data_args(train_data_spec)
        self.train_sets, self.train_xy, self.train_x, self.train_y = read_dataset(train_dataset, train_dataset_args)

    # initialize the activation function
    def init_activation(self):
        self.activation = parse_activation(self.activation_text)

    # parse the arguments to get the values for various variables 
    def parse_config_common(self, arguments):
        if arguments.has_key('corruption_level'):
            self.corruption_levels = [float(arguments['corruption_level']) for n in xrange(100)]
        if arguments.has_key('learning_rate'):
            self.learning_rates = [float(arguments['learning_rate']) for n in xrange(100)]
        if arguments.has_key('batch_size'):
            self.batch_size = int(arguments['batch_size'])
        if arguments.has_key('epoch_number'):
            self.epochs = int(arguments['epoch_number'])
        if arguments.has_key('momentum'):
            self.momentum = float(arguments['momentum'])

        # parse DNN network structure
        nnet_layers = arguments['nnet_spec'].split(':')
        self.n_ins = int(nnet_layers[0])
        self.hidden_layers_sizes = [int(nnet_layers[i]) for i in range(1, len(nnet_layers)-1)]
        self.n_outs = int(nnet_layers[-1])

        # parse pre-training layer number
        self.ptr_layer_number = len(self.hidden_layers_sizes)
        if arguments.has_key('ptr_layer_number'):
            self.ptr_layer_number = int(arguments['ptr_layer_number'])

        # parse activation function
        if arguments.has_key('hidden_activation'):
            self.hidden_activation = parse_activation(arguments['hidden_activation'])
            if arguments['hidden_activation'].startswith('maxout'):
                self.do_maxout = True; self.pool_size = int(arguments['hidden_activation'].replace('maxout:',''))
        if arguments.has_key('1stlayer_reconstruct_activation'):
            self.firstlayer_reconstruct_activation = parse_activation(arguments['1stlayer_reconstruct_activation'])

        # parse sparsity setting
        if arguments.has_key('sparsity'):
            self.sparsity = float(arguments['sparsity'])
        if arguments.has_key('sparsity_weight'):
            self.sparsity_weight = float(arguments['sparsity_weight'])

        # parse various paths for model saving
        if arguments.has_key('cfg_output_file'):
            self.cfg_output_file = arguments['cfg_output_file']
        if arguments.has_key('param_output_file'):
            self.param_output_file = arguments['param_output_file']
        if arguments.has_key('kaldi_output_file'):
            self.kaldi_output_file = arguments['kaldi_output_file']
