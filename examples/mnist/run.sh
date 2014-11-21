#!/bin/bash

wget http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

python data_prep.py

python ../../run_DNN.py --train-data "$working_dir/train.pickle.gz,random=false" \
                        --valid-data "$working_dir/valid.pickle.gz,random=false" \
                        --nnet-spec "$feat_dim:1024:1024:1024:1024:$num_pdfs" \
                        --output-format kaldi --lrate "D:0.1:0.5:0.05,0.05:8" \
                        --wdir $working_dir --output-file $working_dir/dnn.nnet > train.log


