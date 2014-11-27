
import sys
import cPickle
import gzip

if __name__ == '__main__':

    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    cPickle.dump(train_set, gzip.open('train.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(valid_set, gzip.open('valid.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(test_set, gzip.open('test.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)

