# Copyright 2015    Yun Wang      Carnegie Mellon University

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

import os.path
import numpy

# Prepare readers for compressed files
readers = {}
try:
    import gzip
    readers['.gz'] = gzip.GzipFile
except ImportError:
    pass
try:
    import bz2
    readers['.bz2'] = bz2.BZ2File
except ImportError:
    pass

def smart_open(filename, mode = 'rb', *args, **kwargs):
    '''
    Opens a file "smartly":
      * If the filename has a ".gz" or ".bz2" extension, compression is handled
        automatically;
      * If the file is to be read and does not exist, corresponding files with
        a ".gz" or ".bz2" extension will be attempted.
    (The Python packages "gzip" and "bz2" must be installed to deal with the
        corresponding extensions)
    '''
    if 'r' in mode and not os.path.exists(filename):
        for ext in readers:
            if os.path.exists(filename + ext):
                filename += ext
                break
    extension = os.path.splitext(filename)[1]
    return readers.get(extension, open)(filename, mode, *args, **kwargs)

def make_context(feature, left, right):
    '''
    Takes a 2-D numpy feature array, and pads each frame with a specified
        number of frames on either side.
    '''
    feature = [feature]
    for i in range(left):
        feature.append(numpy.vstack((feature[-1][0], feature[-1][:-1])))
    feature.reverse()
    for i in range(right):
        feature.append(numpy.vstack((feature[-1][1:], feature[-1][-1])))
    return numpy.hstack(feature)

def preprocess_feature_and_label(feature, label, opts):
    '''
    Apply the options 'context', 'ignore-label', 'map-label' to the feature
        matrix and label vector.
    '''

    feature = make_context(feature, opts['lcxt'], opts['rcxt'])

    if label is not None:
        if opts.has_key('ignore-label'):
            ignore = opts['ignore-label']
            mask = numpy.array([x not in ignore for x in label])
            feature = feature[mask]
            label = label[mask]
        if opts.has_key('map-label'):
            map = opts['map-label']
            label = numpy.array([map.get(x, x) for x in label])

    return feature, label

def shuffle_feature_and_label(feature, label):
    '''
    Randomly shuffles features and labels in the *same* order.
    '''
    seed = 18877
    numpy.random.seed(seed)
    numpy.random.shuffle(feature)
    numpy.random.seed(seed)
    numpy.random.shuffle(label)

def shuffle_across_partitions(feature_list, label_list):
    '''
    Randomly shuffles features and labels in the same order across partitions.
    '''
    total = sum(len(x) for x in feature_list)
    n = len(feature_list[0])    # Partition size
    buffer = numpy.empty_like(feature_list[0][0])
    seed = 18877
    numpy.random.seed(seed)
    for i in xrange(total - 1, 0, -1):
        j = numpy.random.randint(i + 1)
        buffer[:] = feature_list[i / n][i % n]
        feature_list[i / n][i % n] = feature_list[j / n][j % n]
        feature_list[j / n][j % n] = buffer
        label_list[i / n][i % n], label_list[j / n][j % n] = \
            label_list[j / n][j % n], label_list[i / n][i % n]
