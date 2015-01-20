
import numpy
import sys
import os
import cPickle, gzip

# This script 

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} rbm_feat_file original_dataset new_dataset".format(sys.argv[0])
        print "rbm_feat_file -- path to the RBM feature file"
        print "original_dataset -- the original dataset"
        print "new_dataset -- path to save the new data"
        exit(1)

    rbm_feat_file = sys.argv[1]; original_dataset = sys.argv[2]; new_dataset = sys.argv[3]

    if '.gz' in rbm_feat_file:
        feat_mat = cPickle.load(gzip.open(rbm_feat_file, 'rb'))
    else:
        feat_mat = cPickle.load(open(rbm_feat_file, 'rb'))

    original_feat_mat, labels = cPickle.load(gzip.open(original_dataset, 'rb'))

    assert(feat_mat.shape[0] == original_feat_mat.shape[0])

    cPickle.dump((feat_mat, labels), gzip.open(new_dataset,'wb'), cPickle.HIGHEST_PROTOCOL)


