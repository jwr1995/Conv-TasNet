#!/bin/bash

# Created on 2018/12
# Author: Kaituo XU
# Adapted 2021/02
# Author: William Ravenscroft
source /share/mini1/usr/will/miniconda3/bin/activate
conda init
conda activate cs21

#data=/home/will/data/dummy/cs21
data=/share/mini1/data/audvis/pub/se/mchan/mult/ConferencingSpeech/v1/ConferencingSpeech2021/simulation/data/wavs


stage=1  # Modify this to control to start from which stage
# -- END

dumpdir=data  # directory to put generated json file

# -- START Conv-TasNet Config
train_dir=$dumpdir/train
valid_dir=$dumpdir/dev
evaluate_dir=$dumpdir/eval
separate_dir=$dumpdir/eval
percentage=10 # percetnage of simulated sets to use
sample_rate=16000
segment=6  # seconds
cv_maxlen=6  # seconds
# Network config
N=256
L=70
B=256
H=512
P=3
X=8
R=4
norm_type=gLN
causal=0
mask_nonlinear='relu'
C=1
# Training config
use_cuda=1
id=0
epochs=100
half_lr=1
early_stop=0
max_norm=5
# minibatch
shuffle=1
batch_size=4
num_workers=4
# optimizer
optimizer=adam
lr=1e-3
momentum=0
l2=0
# save and visualize
checkpoint=0
continue_from=""
print_freq=10
visdom=0
visdom_epoch=0
visdom_id="Conv-TasNet Training"
# evaluate
ev_use_cuda=0
cal_sdr=1
corpus=cs21
array=simu_non_uniform
# -- END Conv-TasNet Config

# exp tag
tag="" # tag for managing experiments.

ngpu=1  # always 1

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh


if [ $stage -le 0 ]; then
  echo "Stage 0: Convert sphere format to wav format and generate mixture"
  local/data_prepare.sh --data ${wsj0_origin} --wav_dir ${wsj0_wav}

  echo "NOTE: You should generate mixture by yourself now.
You can use tools/create-speaker-mixtures.zip which is download from
http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip
If you don't have Matlab and want to use Octave, I suggest to replace
all mkdir(...) in create_wav_2speakers.m with system(['mkdir -p '...])
due to mkdir in Octave can not work in 'mkdir -p' way.
e.g.:
mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);
->
system(['mkdir -p ' output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);"
  exit 1
fi


if [ $stage -le 1 ]; then
  echo "Stage 1: Generating json files including wav path and duration"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  preprocess.py --in-dir $data --out-dir $dumpdir --sample-rate $sample_rate --corpus $corpus --percentage $percentage
fi
