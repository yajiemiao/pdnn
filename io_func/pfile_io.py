# Copyright 2013    Yajie Miao    Carnegie Mellon University
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

import gzip
import os
import sys, re
import glob

import struct
import numpy
import theano
import theano.tensor as T
from utils.utils import string2bool
from model_io import log
from io_func import smart_open, preprocess_feature_and_label, shuffle_feature_and_label, shuffle_across_partitions

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
        self.total_frame_num = 0
        self.partition_num = 0
        self.frame_per_partition = 0
        self.end_reading = False

    # read pfile information from the header part
    def read_pfile_info(self):
        line = self.file_read.readline()
        if line.startswith('-pfile_header') == False:
            print "Error: PFile format is wrong, maybe the file was not generated successfully."
            exit(1)
        self.header_size = int(line.split(' ')[-1])
        while (not line.startswith('-end')):
            if line.startswith('-num_sentences'):
                self.num_sentences = int(line.split(' ')[-1])
            elif line.startswith('-num_frames'):
                self.total_frame_num = int(line.split(' ')[-1])
            elif line.startswith('-first_feature_column'):
                self.feat_start_column = int(line.split(' ')[-1])
            elif line.startswith('-num_features'):
                self.original_feat_dim = int(line.split(' ')[-1])
            elif line.startswith('-first_label_column'):
                self.label_start_column = int(line.split(' ')[-1])
            elif line.startswith('-num_labels'):
                self.num_labels = int(line.split(' ')[-1])
            line = self.file_read.readline()
        # partition size in terms of frames
        self.feat_dim = (self.read_opts['lcxt'] + 1 + self.read_opts['rcxt']) * self.original_feat_dim
        self.frame_per_partition = self.read_opts['partition'] / (self.feat_dim * 4)
        batch_residual = self.frame_per_partition % 256
        self.frame_per_partition = self.frame_per_partition - batch_residual

    def read_pfile_data(self):
        # data format for pfile reading
        # s -- sentence index; f -- frame index; d -- features; l -- label
        self.dtype = numpy.dtype({'names': ['d', 'l'],
                                'formats': [('>f', self.original_feat_dim), '>i'],
                                'offsets': [self.feat_start_column * 4, self.label_start_column * 4]})

        # Read the sentence offsets
        self.file_read.seek(self.header_size + 4 * (self.label_start_column + self.num_labels) * self.total_frame_num)
        sentence_offset = struct.unpack(">%di" % (self.num_sentences + 1), self.file_read.read(4 * (self.num_sentences + 1)))

        # Read the data
        self.feat_mats = []
        self.label_vecs = []
        self.file_read.seek(self.header_size)
        for i in xrange(self.num_sentences):
            num_frames = sentence_offset[i+1] - sentence_offset[i]
            if self.file_read is file:  # Not a compressed file
                sentence_array = numpy.fromfile(self.file_read, self.dtype, num_frames)
            else:
                nbytes = 4 * num_frames * (self.label_start_column + self.num_labels)
                d_tmp = self.file_read.read(nbytes)
                sentence_array = numpy.fromstring(d_tmp, self.dtype, num_frames)
            feat_mat = numpy.asarray(sentence_array['d'])
            label_vec = numpy.asarray(sentence_array['l'])
            feat_mat, label_vec = preprocess_feature_and_label(feat_mat, label_vec, self.read_opts)
            if len(self.feat_mats) > 0 and read_frames < self.frame_per_partition:
                num_frames = min(len(feat_mat), self.frame_per_partition - read_frames)
                self.feat_mats[-1][read_frames : read_frames + num_frames] = feat_mat[:num_frames]
                self.label_vecs[-1][read_frames : read_frames + num_frames] = label_vec[:num_frames]
                feat_mat = feat_mat[num_frames:]
                label_vec = label_vec[num_frames:]
                read_frames += num_frames
            if len(feat_mat) > 0:
                read_frames = len(feat_mat)
                self.feat_mats.append(numpy.zeros((self.frame_per_partition, self.feat_dim), dtype = theano.config.floatX))
                self.label_vecs.append(numpy.zeros(self.frame_per_partition, dtype = numpy.int32))
                self.feat_mats[-1][:read_frames] = feat_mat
                self.label_vecs[-1][:read_frames] = label_vec
        # finish reading; close the file
        self.file_read.close()
        self.feat_mats[-1] = self.feat_mats[-1][:read_frames]
        self.label_vecs[-1] = self.label_vecs[-1][:read_frames]
        if self.read_opts['random']:
            shuffle_across_partitions(self.feat_mats, self.label_vecs)
        self.partition_num = len(self.feat_mats)
        self.partition_index = 0

    def load_next_partition(self, shared_xy):
        feat = self.feat_mats[self.partition_index]
        label = self.label_vecs[self.partition_index]
        shared_x, shared_y = shared_xy

        shared_x.set_value(feat.astype(theano.config.floatX), borrow = True)
        shared_y.set_value(label.astype(theano.config.floatX), borrow = True)

        self.cur_frame_num = len(feat)
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
            self.feat_mats = []
            self.label_vecs = []
            self.read_pfile_info()
            self.read_pfile_data()
        self.end_reading = False
        self.partition_index = 0

    def make_shared(self):
        # define shared variables
        feat = self.feat_mats[0]
        label = self.label_vecs[0].astype(theano.config.floatX)

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

        # store features and labels for each data partition
        self.feat = numpy.zeros((10, self.feat_dim), dtype = theano.config.floatX)
        self.label = numpy.zeros(10, dtype = numpy.int32)

        self.end_reading = False

    # read pfile information from the header part and the sentence index
    def read_pfile_info(self):
        line = self.file_read.readline()
        if line.startswith('-pfile_header') == False:
            print "Error: PFile format is wrong, maybe the file was not generated successfully."
            exit(1)
        self.header_size = int(line.split(' ')[-1])
        while (not line.startswith('-end')):
            if line.startswith('-num_sentences'):
                self.num_sentences = int(line.split(' ')[-1])
            elif line.startswith('-num_frames'):
                self.total_frame_num = int(line.split(' ')[-1])
            elif line.startswith('-first_feature_column'):
                self.feat_start_column = int(line.split(' ')[-1])
            elif line.startswith('-num_features'):
                self.original_feat_dim = int(line.split(' ')[-1])
            elif line.startswith('-first_label_column'):
                self.label_start_column = int(line.split(' ')[-1])
            elif line.startswith('-num_labels'):
                self.num_labels = int(line.split(' ')[-1])
            line = self.file_read.readline()
        # partition size in terms of frames
        self.feat_dim = (self.read_opts['lcxt'] + 1 + self.read_opts['rcxt']) * self.original_feat_dim
        self.frame_per_partition = self.read_opts['partition'] / (self.feat_dim * 4)
        batch_residual = self.frame_per_partition % 256
        self.frame_per_partition = self.frame_per_partition - batch_residual
        # read the sentence offsets
        self.file_read.seek(self.header_size + 4 * (self.label_start_column + self.num_labels) * self.total_frame_num)
        self.sentence_offset = struct.unpack(">%di" % (self.num_sentences + 1), self.file_read.read(4 * (self.num_sentences + 1)))

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
        self.sentence_index = 0

        if first_time_reading:
            self.feat = numpy.zeros((self.frame_per_partition, self.feat_dim), dtype = theano.config.floatX)
            self.label = numpy.zeros(self.frame_per_partition, dtype = numpy.int32)

        self.feat_buffer = None
        self.label_buffer = None

    # load the n-th (0 indexed) partition to the GPU memory
    def load_next_partition(self, shared_xy):
        shared_x, shared_y = shared_xy

        # read one partition from disk; data format for pfile reading
        # d -- features; l -- label
        self.dtype = numpy.dtype({'names': ['d', 'l'],
                                'formats': [('>f', self.original_feat_dim), '>i'],
                                'offsets': [self.feat_start_column * 4, self.label_start_column * 4]})

        if self.feat_buffer is None:
            read_frames = 0
        else:   # An utterance hasn't been completely consumed yet
            read_frames = min(self.frame_per_partition, len(self.feat_buffer))
            self.feat[0:read_frames] = self.feat_buffer[0:read_frames]
            self.label[0:read_frames] = self.label_buffer[0:read_frames]
            if read_frames == len(self.feat_buffer):
                self.feat_buffer = None
                self.label_buffer = None
            else:
                self.feat_buffer = self.feat_buffer[read_frames:]
                self.label_buffer = self.label_buffer[read_frames:]

        while read_frames < self.frame_per_partition and self.sentence_index < self.num_sentences:
            num_frames = self.sentence_offset[self.sentence_index + 1] - self.sentence_offset[self.sentence_index]
            if self.file_read is file:  # Not a compressed file
                sentence_array = numpy.fromfile(self.file_read, self.dtype, num_frames)
            else:
                nbytes = 4 * num_frames * (self.label_start_column + self.num_labels)
                d_tmp = self.file_read.read(nbytes)
                sentence_array = numpy.fromstring(d_tmp, self.dtype, num_frames)
            feat_mat = numpy.asarray(sentence_array['d'])
            label_vec = numpy.asarray(sentence_array['l'])
            feat_mat, label_vec = preprocess_feature_and_label(feat_mat, label_vec, self.read_opts)
            num_frames = len(feat_mat)

            if read_frames + num_frames > self.frame_per_partition:
                # Utterance won't fit in current partition, use some frames and keep the rest for the next partition
                num_frames = self.frame_per_partition - read_frames
                self.feat_buffer = feat_mat[num_frames:]
                self.label_buffer = label_vec[num_frames:]
                feat_mat = feat_mat[:num_frames]
                label_vec = label_vec[:num_frames]

            self.feat[read_frames:(read_frames + num_frames)] = feat_mat
            self.label[read_frames:(read_frames + num_frames)] = label_vec
            read_frames += num_frames
            self.sentence_index += 1

        if self.read_opts['random']:
            shuffle_feature_and_label(self.feat[:read_frames], self.label[:read_frames])

        shared_x.set_value(self.feat, borrow = True)
        shared_y.set_value(self.label, borrow = True)
        self.cur_frame_num = read_frames

        if self.sentence_index >= self.num_sentences and self.feat_buffer is None:
            # move on to the next pfile
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
