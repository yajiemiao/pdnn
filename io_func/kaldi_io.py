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

import gzip
import os
import sys, re
import glob
import struct

import numpy
import theano
import theano.tensor as T
from model_io import log
from io_func import smart_open, preprocess_feature_and_label, shuffle_feature_and_label

class KaldiDataRead(object):

    def __init__(self, scp_list = [], read_opts = None):

        self.scp_file = scp_list[0]     # path to the .scp file
        self.read_opts = read_opts
        if read_opts.has_key('label'):
            self.ali_file = read_opts['label']  # path to the alignment file
            self.ali_provided = True    # if alignment is provided
        else:
            self.ali_file = ''
            self.ali_provided = False

        self.scp_file_read = None
        self.scp_cur_pos = None
        # feature information
        self.original_feat_dim = 1024
        self.feat_dim = 1024
        self.cur_frame_num = 1024
        self.alignment = {}

        # store features and labels for each data partition
        self.feat = numpy.zeros((10, self.feat_dim), dtype = theano.config.floatX)
        self.label = numpy.zeros(10, dtype = numpy.int32)

    # read the alignment of all the utterances and keep the alignment in CPU memory.
    def read_alignment(self):
        f_read = smart_open(self.ali_file, 'r')
        for line in f_read:
            line = line.replace('\n','').strip()
            if len(line) < 1: # this is an empty line, skip
                continue
            [utt_id, utt_ali] = line.split(' ', 1)
            # this utterance has empty alignment, skip
            if len(utt_ali) < 1:
                continue
            self.alignment[utt_id] = numpy.fromstring(utt_ali, dtype=numpy.int32, sep=' ')
        f_read.close()

    # read the feature matrix of the next utterance
    def read_next_utt(self):
        self.scp_cur_pos = self.scp_file_read.tell()
        next_scp_line = self.scp_file_read.readline()
        if next_scp_line == '' or next_scp_line == None:    # we are reaching the end of one epoch
            return '', None
        utt_id, path_pos = next_scp_line.replace('\n','').split(' ')
        path, pos = path_pos.split(':')

        ark_read_buffer = smart_open(path, 'rb')
        ark_read_buffer.seek(int(pos),0)

        # now start to read the feature matrix into a numpy matrix
        header = struct.unpack('<xcccc', ark_read_buffer.read(5))
        if header[0] != "B":
            print "Input .ark file is not binary"; exit(1)

        rows = 0; cols= 0
        m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
        n, cols = struct.unpack('<bi', ark_read_buffer.read(5))

        tmp_mat = numpy.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=numpy.float32)
        utt_mat = numpy.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

        return utt_id, utt_mat

    def is_finish(self):
        return self.end_reading

    def initialize_read(self, first_time_reading = False):
        self.scp_file_read = smart_open(self.scp_file, 'r')
        if first_time_reading:
            utt_id, utt_mat = self.read_next_utt()
            self.original_feat_dim = utt_mat.shape[1]
            self.scp_file_read = smart_open(self.scp_file, 'r')

            # compute the feature dimension
            self.feat_dim = (self.read_opts['lcxt'] + 1 + self.read_opts['rcxt']) * self.original_feat_dim

            # allocate the feat matrix according to the specified partition size
            self.max_frame_num = self.read_opts['partition'] / (self.feat_dim * 4)
            self.feats = numpy.zeros((self.max_frame_num, self.feat_dim), dtype = theano.config.floatX)
            if self.ali_provided:
                self.read_alignment()
                self.labels = numpy.zeros(self.max_frame_num, dtype = numpy.int32)

        self.end_reading = False
        self.feat_buffer = None
        self.label_buffer = None

    # load the n-th (0 indexed) partition to the GPU memory
    def load_next_partition(self, shared_xy):
        shared_x, shared_y = shared_xy

        if self.feat_buffer is None:
            read_frame_num = 0
        else:   # An utterance hasn't been completely consumed yet
            read_frame_num = min(self.max_frame_num, len(self.feat_buffer))
            self.feats[0:read_frame_num] = self.feat_buffer[0:read_frame_num]
            if self.ali_provided:
                self.labels[0:read_frame_num] = self.label_buffer[0:read_frame_num]
            if read_frame_num == len(self.feat_buffer):
                self.feat_buffer = None
                self.label_buffer = None
            else:
                self.feat_buffer = self.feat_buffer[read_frame_num:]
                if self.ali_provided:
                    self.label_buffer = self.label_buffer[read_frame_num:]

        while read_frame_num < self.max_frame_num:
            utt_id, utt_mat = self.read_next_utt()
            if utt_id == '':    # No more utterances available
                self.end_reading = True
                break
            if self.ali_provided and (self.alignment.has_key(utt_id) is False):
                continue
            rows = len(utt_mat)

            if self.ali_provided:
                ali_utt = self.alignment[utt_id]
                if len(ali_utt) != rows:
                    continue
            else:
                ali_utt = None

            utt_mat, ali_utt = preprocess_feature_and_label(utt_mat, ali_utt, self.read_opts)
            rows = len(utt_mat)

            if read_frame_num + rows > self.max_frame_num:
                # Utterance won't fit in current partition, use some frames and keep the rest for the next partition
                rows = self.max_frame_num - read_frame_num
                self.feat_buffer = utt_mat[rows:]
                utt_mat = utt_mat[:rows]
                if self.ali_provided:
                    self.label_buffer = ali_utt[rows:]
                    ali_utt = ali_utt[:rows]

            self.feats[read_frame_num:(read_frame_num + rows)] = utt_mat
            if self.ali_provided:
                self.labels[read_frame_num:(read_frame_num + rows)] = ali_utt
            read_frame_num += rows

        if self.read_opts['random']:
            shuffle_feature_and_label(self.feats[0:read_frame_num], self.labels[0:read_frame_num])

        shared_x.set_value(self.feats[0:read_frame_num], borrow=True)
        if self.ali_provided:
            shared_y.set_value(self.labels[0:read_frame_num], borrow=True)
        self.cur_frame_num = read_frame_num

    def make_shared(self):
        shared_x = theano.shared(self.feats, name = 'x', borrow = True)
        shared_y = theano.shared(self.labels, name = 'y', borrow = True)
        return shared_x, shared_y
