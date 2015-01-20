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

# train the RBM model
echo "Training the RBM model ..."
python $pdnndir/cmds/run_RBM.py --train-data "train.pickle.gz" \
                                --nnet-spec "784:1024:1024:10" --wdir ./ \
                                --epoch-number 10 --batch-size 128 --first_layer_type gb \
                                --ptr-layer-number 2 \
                                --param-output-file rbm.param --cfg-output-file rbm.cfg  >& rbm.training.log || exit 1;

# generate feature representations with the RBM model
for set in train valid test; do
  echo "Generating RBM features on the $set set ..."
  python $pdnndir/cmds/run_Extract_Feats.py --data "$set.pickle.gz" \
                                          --nnet-param rbm.param --nnet-cfg rbm.cfg \
                                          --output-file "rfeat.pickle.gz" --layer-index 1 \
                                          --batch-size 100 >& rfeat.$set.log || exit 1;

  # create the new set with the new RBM features and the original labels
  python create_new_datasets.py "rfeat.pickle.gz" "$set.pickle.gz" "new${set}.pickle.gz" || exit 1;
  rm -rf rfeat.pickle.gz
done

# train a DNN model over the new features
echo "Training the DNN model ..."
python $pdnndir/cmds/run_DNN.py --train-data "newtrain.pickle.gz" \
                                --valid-data "newvalid.pickle.gz" \
                                --nnet-spec "1024:1024:1024:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:200" --model-save-step 20 \
                                --param-output-file dnn.param --cfg-output-file dnn.cfg  >& dnn.training.log || exit 1;

# classification on the test set
echo "Do classification on the test set ..."
python $pdnndir/cmds/run_Extract_Feats.py --data "newtest.pickle.gz" \
                                          --nnet-param dnn.param --nnet-cfg dnn.cfg \
                                          --output-file "dnn.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 >& dnn.testing.log || exit 1;

python show_results.py dnn.classify.pickle.gz
