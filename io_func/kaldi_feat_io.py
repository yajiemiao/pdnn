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

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano

# Functions to read and write Kaldi text-formatted .scp and .ark
# Only used for generating convolution layer activation

class KaldiReadIn(object):

    def __init__(self, ark_path):

        self.ark_path = ark_path
        self.infile = None

    def open(self):
        if self.ark_path.find('.gz') != -1:
            self.infile = gzip.open(self.ark_path,"r")
        else:
            self.infile = open(self.ark_path,"r")

    def next(self):
        content = []
        line = self.infile.readline()
        # already at the final line
        if line == '':
            return '', None
        elements = line.split(' ')
        uttID = elements[0]
        while 1:
            line = self.infile.readline()
            if line.find(' ]') != -1:
                break
            content.append(line.strip())
        content.append(line.strip().replace(' ]', ''))

        num_row = len(content)
        num_col = len(content[0].split(' '))
        
        feat_mat = numpy.ndarray(shape=(num_row,num_col), dtype=theano.config.floatX)
        for i in xrange(num_row):
            elements = content[i].split(' ')
            for j in xrange(num_col):
                feat_mat[i,j] = float(elements[j])
        return uttID, feat_mat

    def close(self):
        self.infile.close()

class KaldiWriteOut(object):

    def __init__(self, scp_path, ark_path):

        self.ark_path = ark_path
        self.scp_path = scp_path
        self.out_ark = None
        self.out_scp = None

        self.offset = 0

    def open(self):
        self.out_ark = open(self.ark_path,"a")
        self.out_scp = open(self.scp_path,"a")


    def write(self, uttID, data):
        start_offset = self.offset + len(uttID + ' ')
        line = uttID + '  [\n'
        self.out_ark.write(line)
        self.offset += len(line)      
 
        num_row, num_col = data.shape
        # write out ark 
        for i in xrange(num_row):
            line = ' '
            for j in xrange(num_col):
                line = line + ' ' + str(data[i,j])
            if i == (num_row - 1):
                line += ' ]'
            line += '\n'
            self.out_ark.write(line)
            self.offset += len(line)

        # write out scp
        scp_out = uttID + ' ' + self.ark_path + ':' + str(start_offset)
        self.out_scp.write(scp_out + '\n')

    def close(self):
        self.out_ark.close()
        self.out_scp.close()
