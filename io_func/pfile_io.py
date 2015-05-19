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

import gzip
import os
import sys, re
import glob

import numpy
import theano
import theano.tensor as T
from utils.utils import string_2_bool
from model_io import log
from io_func import smart_open

class PfileDataRead(object):

    def __init__(self, pfile_path_list, read_opts):

        self.pfile_path_list = pfile_path_list
        self.cur_pfile_index = 0
        self.pfile_path = pfile_path_list[0]
        self.read_opts = read_opts

        # pfile information
        self.header_size = 32768
        self.feat_start_column = 2
        self.feat_dim = 1024
        self.label_start_column = 442
        self.num_labels = 1

        # markers while reading data
        self.frame_to_read = 0
        self.partition_num = 0
        self.frame_per_partition = 0

        # store number of frames, features and labels for each data partition
        self.frame_nums = []
        self.feat_mats = []
        self.label_vecs = []

        # other variables to be consistent with PfileDataReadStream
        self.partition_index = 0
        self.cur_frame_num = 0
        self.end_reading = False

    # read pfile information from the header part
    def read_pfile_info(self):
        line = self.file_read.readline()
        if line.startswith('-pfile_header') == False:
            print "Error: PFile format is wrong, maybe the file was not generated successfully."
            exit(1)
        self.header_size = int(line.split(' ')[-1])
        while (not line.startswith('-end')):
            if line.startswith('-num_frames'):
                self.frame_to_read = int(line.split(' ')[-1])
            elif line.startswith('-first_feature_column'):
                self.feat_start_column = int(line.split(' ')[-1])
            elif line.startswith('-num_features'):
                self.feat_dim = int(line.split(' ')[-1])
            elif line.startswith('-first_label_column'):
                self.label_start_column = int(line.split(' ')[-1])
            elif line.startswith('-num_labels'):
                self.num_labels = int(line.split(' ')[-1])
            line = self.file_read.readline()
        # partition size in terms of frames
        self.frame_per_partition = self.read_opts['partition'] / (self.feat_dim * 4)
        batch_residual = self.frame_per_partition % 256
        self.frame_per_partition = self.frame_per_partition - batch_residual

    def read_pfile_data(self):
        # data format for pfile reading
        # s -- sentence index; f -- frame index; d -- features; l -- label
        self.dtype = numpy.dtype({'names': ['d', 'l'],
                                'formats': [('>f', self.feat_dim), '>i'],
                                'offsets': [self.feat_start_column * 4, (self.feat_start_column + self.feat_dim) * 4]})
        # Now we skip the file header
        self.file_read.seek(self.header_size, 0)
        while True:
            if self.frame_to_read == 0:
                break
            frameNum_this_partition = min(self.frame_to_read, self.frame_per_partition)
            if self.pfile_path.endswith('.gz'):
                nbytes = 4 * self.frame_per_partition * (self.label_start_column + self.num_labels)
                d_tmp = self.file_read.read(nbytes)
                partition_array = numpy.fromstring(d_tmp, self.dtype, frameNum_this_partition)
            else:
                partition_array = numpy.fromfile(self.file_read, self.dtype, frameNum_this_partition)
            feat_mat = numpy.asarray(partition_array['d'], dtype = theano.config.floatX)
            label_vec = numpy.asarray(partition_array['l'], dtype = theano.config.floatX)
            self.feat_mats.append(feat_mat)
            self.label_vecs.append(label_vec)
            self.frame_nums.append(len(label_vec))
            self.frame_to_read = self.frame_to_read - frameNum_this_partition
        # finish reading; close the file
        self.partition_num = len(self.feat_mats)
        self.file_read.close()

    def load_next_partition(self, shared_xy):
        feat = self.feat_mats[self.partition_index]
        label = self.label_vecs[self.partition_index]
        shared_x, shared_y = shared_xy

        if self.read_opts['random']:  # randomly shuffle features and labels in the *same* order
            numpy.random.seed(18877)
            numpy.random.shuffle(feat)
            numpy.random.seed(18877)
            numpy.random.shuffle(label)
        shared_x.set_value(feat, borrow=True)
        shared_y.set_value(label, borrow=True)

        self.cur_frame_num = self.frame_nums[self.partition_index]
        self.partition_index = self.partition_index + 1
        if self.partition_index >= self.partition_num:
            self.partition_index = 0
            self.cur_pfile_index += 1
            if self.cur_pfile_index >= len(self.pfile_path_list):   # the end of one epoch
                self.end_reading = True
                self.cur_pfile_index = 0
            else:
                self.initialize_read()

    def is_finish(self):
        return self.end_reading

    # reopen pfile with the same filename
    def reopen_file(self):
        self.file_read = smart_open(self.pfile_path, 'rb')
        self.read_pfile_info()
        self.initialize_read()
        self.read_pfile_data()

    def initialize_read(self, first_time_reading = False):
        pfile_path = self.pfile_path_list[self.cur_pfile_index]
        self.file_read = smart_open(pfile_path, 'rb')

        if first_time_reading or len(self.pfile_path_list) > 1:
            self.frame_nums = []
            self.feat_mats = []
            self.label_vecs = []
            self.read_pfile_info()
            self.read_pfile_data()
        self.end_reading = False
        self.partition_index = 0

    def make_shared(self):
        # define shared variables
        feat = self.feat_mats[0]
        label = self.label_vecs[0]

        if self.read_opts['random']:  # randomly shuffle features and labels in the *same* order
            numpy.random.seed(18877)
            numpy.random.shuffle(feat)
            numpy.random.seed(18877)
            numpy.random.shuffle(label)

        shared_x = theano.shared(feat, name = 'x', borrow = True)
        shared_y = theano.shared(label, name = 'y', borrow = True)
        return shared_x, shared_y

class PfileDataReadStream(object):

    def __init__(self, pfile_path_list, read_opts):

        self.pfile_path_list = pfile_path_list
        self.cur_pfile_index = 0
        self.read_opts = read_opts

        # pfile information
        self.header_size = 32768
        self.feat_start_column = 2
        self.feat_dim = 1024
        self.total_frame_num = 0
        self.label_start_column = 442
        self.num_labels = 1

        # markers while reading data
        self.frame_to_read = 0
        self.partition_num = 0
        self.frame_per_partition = 0

        # store number of frames, features and labels for each data partition
        self.feat = numpy.zeros((10,self.feat_dim), dtype=theano.config.floatX)
        self.label = numpy.zeros((10,), dtype=theano.config.floatX)
        self.cur_frame_num = 0
        self.end_reading = False

    # read pfile information from the header part
    def read_pfile_info(self):
        line = self.file_read.readline()
        if line.startswith('-pfile_header') == False:
            print "Error: PFile format is wrong, maybe the file was not generated successfully."
            exit(1)
        self.header_size = int(line.split(' ')[-1])
        while (not line.startswith('-end')):
            if line.startswith('-num_frames'):
                self.total_frame_num = self.frame_to_read = int(line.split(' ')[-1])
            elif line.startswith('-first_feature_column'):
                self.feat_start_column = int(line.split(' ')[-1])
            elif line.startswith('-num_features'):
                self.feat_dim = int(line.split(' ')[-1])
            elif line.startswith('-first_label_column'):
                self.label_start_column = int(line.split(' ')[-1])
            elif line.startswith('-num_labels'):
                self.num_labels = int(line.split(' ')[-1])
            line = self.file_read.readline()
        # partition size in terms of frames
        self.frame_per_partition = self.read_opts['partition'] / (self.feat_dim * 4)
        batch_residual = self.frame_per_partition % 256
        self.frame_per_partition = self.frame_per_partition - batch_residual

    # reopen pfile with the same filename
    def reopen_file(self):
        self.file_read = smart_open(self.pfile_path, 'rb')
        self.read_pfile_info()
        self.initialize_read()

    def is_finish(self):
        return self.end_reading

    def initialize_read(self, first_time_reading = False):
        self.pfile_path = self.pfile_path_list[self.cur_pfile_index]
        self.file_read = smart_open(self.pfile_path)
        self.read_pfile_info()
        self.end_reading = False

        self.file_read.seek(self.header_size, 0)
        self.frame_to_read = self.total_frame_num

    # load the n-th (0 indexed) partition to the GPU memory
    def load_next_partition(self, shared_xy):
        shared_x, shared_y = shared_xy

        # read one partition from disk; data format for pfile reading
        # d -- features; l -- label
        self.dtype = numpy.dtype({'names': ['d', 'l'],
                                'formats': [('>f', self.feat_dim), '>i'],
                                'offsets': [self.feat_start_column * 4, (self.feat_start_column + self.feat_dim) * 4]})
        if self.feat is None:  # haven't read anything, then skip the file header
            self.file_read.seek(self.header_size, 0)

        frameNum_this_partition = min(self.frame_to_read, self.frame_per_partition)
        if self.pfile_path.endswith('.gz'):
            nbytes = 4 * self.frame_per_partition * (self.label_start_column + self.num_labels)
            d_tmp = self.file_read.read(nbytes)
            partition_array = numpy.fromstring(d_tmp, self.dtype, frameNum_this_partition)
        else:
            partition_array = numpy.fromfile(self.file_read, self.dtype, frameNum_this_partition)
        self.feat = numpy.asarray(partition_array['d'], dtype = theano.config.floatX)
        self.label = numpy.asarray(partition_array['l'], dtype = theano.config.floatX)
        self.cur_frame_num = frameNum_this_partition
        self.frame_to_read = self.frame_to_read - frameNum_this_partition

        if self.read_opts['random']:  # randomly shuffle features and labels in the *same* order
            numpy.random.seed(18877)
            numpy.random.shuffle(self.feat)
            numpy.random.seed(18877)
            numpy.random.shuffle(self.label)

        shared_x.set_value(self.feat, borrow=True)
        shared_y.set_value(self.label, borrow=True)

        # move on to the next pfile
        if self.frame_to_read <= 0:
            self.cur_pfile_index += 1
            if self.cur_pfile_index >= len(self.pfile_path_list):
                self.end_reading = True
                self.cur_pfile_index = 0
            else:
                self.initialize_read()

    def make_shared(self):
        shared_x = theano.shared(self.feat, name = 'x', borrow = True)
        shared_y = theano.shared(self.label, name = 'y', borrow = True)
        return shared_x, shared_y
