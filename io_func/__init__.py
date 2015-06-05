import os.path
import gzip
import bz2
import numpy

def smart_open(filename, mode = 'rb', *args, **kwargs):
    '''
    Opens a file "smartly":
      * If the filename has a ".gz" or ".bz2" extension, compression is handled
        automatically;
      * If the file is to be read and does not exist, corresponding files with
        a ".gz" or ".bz2" extension will be attempted.
    '''
    readers = {'.gz': gzip.GzipFile, '.bz2': bz2.BZ2File}
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
