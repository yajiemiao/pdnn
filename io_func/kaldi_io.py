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

import gzip
import os
import sys, re
import glob
import struct

import numpy
import theano
import theano.tensor as T
from model_io import log
from io_func import smart_open

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

        # left and right context for feature splicing
        self.left_context = 0; self.right_context = 0
        if read_opts.has_key('lcxt'):
            self.left_context = int(read_opts['lcxt'])
        if read_opts.has_key('rcxt'):
            self.right_context = int(read_opts['rcxt'])

        self.scp_file_read = None
        self.scp_cur_pos = None
        # feature information
        self.original_feat_dim = 1024
        self.feat_dim = 1024
        self.cur_frame_num = 1024
        self.alignment = {}

        self.max_frame_num = 0

        # store features and labels for each data partition
        self.feats = numpy.zeros((10,self.feat_dim), dtype=theano.config.floatX)
        self.labels = numpy.zeros((10,), dtype=numpy.int32)

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
            self.feat_dim = self.original_feat_dim
            if self.left_context > 0 or self.right_context > 0:
                self.feat_dim = (self.left_context + 1 + self.right_context) * self.original_feat_dim

            # allocate the feat matrix according to the specified partition size
            self.max_frame_num = self.read_opts['partition'] / (self.feat_dim * 4)
            self.feats = numpy.zeros((self.max_frame_num, self.feat_dim), dtype=theano.config.floatX)
            if self.ali_provided:
                self.read_alignment()
                self.labels = numpy.zeros((self.max_frame_num,), dtype=numpy.int32)
            print self.original_feat_dim, self.feat_dim
        self.end_reading = False

    # make context of a numpy matrix
    def make_context_matrix(self, mat):
        rows, cols = mat.shape
        cxt_size = self.left_context + 1 + self.right_context
        cols_cxt = cxt_size * cols
        mat_cxt = numpy.zeros((rows, cols_cxt), dtype=theano.config.floatX)
        for t in xrange(rows):
            for j in xrange(cxt_size):
                t2 = t + j - self.left_context
                if t2 < 0:
                    t2 = 0
                if t2 >= rows:
                    t2 = rows - 1
                mat_cxt[t, j*cols:(j+1)*cols] = mat[t2]
        return mat_cxt

    # load the n-th (0 indexed) partition to the GPU memory
    def load_next_partition(self, shared_xy):
        shared_x, shared_y = shared_xy
        read_frame_num = 0
        while True:
            utt_id, utt_mat = self.read_next_utt()
            if utt_id == '':
                self.end_reading = True
                break
            if self.ali_provided and (self.alignment.has_key(utt_id) is False):
                continue
            rows, cols = utt_mat.shape

            ali_utt = None
            if self.ali_provided:
                ali_utt = self.alignment[utt_id]
                if ali_utt.shape[0] != rows:
                    continue
            if read_frame_num + rows > self.max_frame_num:
                self.scp_file_read.seek(self.scp_cur_pos)
                break
            else:
                if self.left_context > 0 or self.right_context > 0:
                    utt_mat = self.make_context_matrix(utt_mat)
                self.feats[read_frame_num:(read_frame_num+rows)] = utt_mat
                if self.ali_provided:
                    self.labels[read_frame_num:(read_frame_num+rows)] = ali_utt
                read_frame_num += rows

        if self.read_opts['random']:  # randomly shuffle features and labels in the *same* order
            numpy.random.seed(18877)
            numpy.random.shuffle(self.feats[0:read_frame_num])
            if self.ali_provided:
                numpy.random.seed(18877)
                numpy.random.shuffle(self.labels[0:read_frame_num])

        shared_x.set_value(self.feats[0:read_frame_num], borrow=True)
        if self.ali_provided:
            shared_y.set_value(self.labels[0:read_frame_num], borrow=True)
        self.cur_frame_num = read_frame_num

    def make_shared(self):
        shared_x = theano.shared(self.feats, name = 'x', borrow = True)
        shared_y = theano.shared(self.labels, name = 'y', borrow = True)
        return shared_x, shared_y
