#!/bin/bash

# two variables you need to set
pdnndir=/data/ASR5/babel/ymiao/tools/pdnn  # pointer to PDNN
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

# download mnist dataset
wget http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

# split the dataset to training, validation and testing sets
# you will see train.pickle.gz, valid.pickle.gz, test.pickle.gz
echo "Preparing datasets ..."
python data_prep.py

# train DNN model
echo "Training the DNN model ..."
python $pdnndir/cmds/run_DNN.py --train-data "train.pickle.gz" \
                                --valid-data "valid.pickle.gz" \
                                --nnet-spec "784:1024:1024:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.01:500" \
                                --param-output-file dnn.param --cfg-output-file dnn.cfg  >& dnn.training.log

# classification on the testing data; -1 means the final layer, that is, the classification softmax layer
echo "Classifying the testing data ..."
python $pdnndir/cmds/run_Extract_Feats.py --data "test.pickle.gz" \
                                          --nnet-param dnn.param --nnet-cfg dnn.cfg \
                                          --output-file "dnn.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 >& dnn.testing.log

python show_results.py dnn.classify.pickle.gz

