#!/bin/bash

# Created on 2018/12
# Author: Kaituo XU
# Adapted 2021/02
# Author: William Ravenscroft
source /share/mini1/usr/will/miniconda3/bin/activate
conda init
conda activate cs21
#export LDLIBRARYPATH=/share/mini1/sw/std/cuda/
#_LIBRARY_PATH=${LD_LIBRARY_PATH}:/share/mini1/sw/std/cuda/cuda10.1/x86_64/lib64/:/share/mini1/sw/std/cuda/cuda10.1/x86_64/include/:/share/mini1/sw/std/cuda/cuda10.1/cuda/:/share/mini1/sw/std/cuda/cuda10.1/x86_64/lib64/stubs

#data=/home/will/data/dummy/cs21
data=/share/mini1/data/audvis/pub/se/mchan/mult/ConferencingSpeech/v1/eval_data/task1/
stage=1  # Modify this to control to start from which stage

dumpdir=data  # directory to put generated json file

# -- START Conv-TasNet Config
train_dir=$dumpdir/train
valid_dir=$dumpdir/dev
evaluate_dir=$dumpdir/eval
separate_dir=$dumpdir/eval
percentage=8
sample_rate=16000
segment=2  # seconds
cv_maxlen=3   # seconds

# Training config
use_cuda=1
id=0,1,2,3

# minibatch
shuffle=1
batch_size=4
num_workers=4
# optimizer
optimizer=adam
lr=5e-3
momentum=0
l2=0
# save and visualize
checkpoint=1
continue_from=""
print_freq=10
visdom=0
visdom_epoch=0
visdom_id="Conv-TasNet Training"
# evaluate
ev_use_cuda=1
cal_sdr=1
figures=True

#corpus params
corpus=cs21
array=simu_non_uniform
multichannel=True

# -- END Conv-TasNet Config

# exp tag
tag="BIG" # tag for managing experiments.

ngpu=1  # always 1

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh


if [ $stage -le 1 ]; then
  echo "Stage 1: Generating json files including wav path and duration"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  preprocess_eval.py --in-dir $data --out-dir $dumpdir --sample-rate $sample_rate --corpus $corpus --percentage $percentage
fi
