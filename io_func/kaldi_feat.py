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

import struct

from io_func import smart_open

# Classes to read and write Kaldi features. They are used when PDNN passes Kaldi features
# through trained models and saves network activation into Kaldi features. Currently we
# are using them during decoding of convolutional networks.

# Class to read Kaldi features. Each time, it reads one line of the .scp file
# and reads in the corresponding features into a numpy matrix. It only supports
# binary-formatted .ark files. Text and compressed .ark files are not supported.
class KaldiReadIn(object):

    def __init__(self, scp_path):

        self.scp_path = scp_path
        self.scp_file_read = smart_open(self.scp_path,"r")

    def read_next_utt(self):
        next_scp_line = self.scp_file_read.readline()
        if next_scp_line == '' or next_scp_line == None:
            return '', None
        utt_id, path_pos = next_scp_line.replace('\n','').split(' ')
        path, pos = path_pos.split(':')

        ark_read_buffer = smart_open(path, 'rb')
        ark_read_buffer.seek(int(pos),0)
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


# Class to write numpy matrix into Kaldi .ark file. It only supports binary-formatted .ark files.
# Text and compressed .ark files are not supported.
class KaldiWriteOut(object):

    def __init__(self, ark_path):

        self.ark_path = ark_path
        self.ark_file_write = smart_open(ark_path,"wb")

    def write_kaldi_mat(self, utt_id, utt_mat):
        utt_mat = numpy.asarray(utt_mat, dtype=numpy.float32)
        rows, cols = utt_mat.shape
        self.ark_file_write.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
        self.ark_file_write.write(struct.pack('<cxcccc', ' ','B','F','M',' '))
        self.ark_file_write.write(struct.pack('<bi', 4, rows))
        self.ark_file_write.write(struct.pack('<bi', 4, cols))
        self.ark_file_write.write(utt_mat)

    def close(self):
        self.ark_file_write.close()
