#!/bin/bash

# Created on 2018/12
# Author: Kaituo XU
# Adapted 2021/02
# Author: William Ravenscroft
source /share/mini1/usr/will/miniconda3/bin/activate
conda init

if [[ $(hostname) = node27 ]]
then
  conda activate torch11
  echo "Using environment: torch11"
else
  conda activate cs21
  echo "Using environment: cs21"
fi

#export LDLIBRARYPATH=/share/mini1/sw/std/cuda/
#_LIBRARY_PATH=${LD_LIBRARY_PATH}:/share/mini1/sw/std/cuda/cuda10.1/x86_64/lib64/:/share/mini1/sw/std/cuda/cuda10.1/x86_64/include/:/share/mini1/sw/std/cuda/cuda10.1/cuda/:/share/mini1/sw/std/cuda/cuda10.1/x86_64/lib64/stubs

export NCCL_SOCKET_IFNAME=virbr0
export NCCL_IB_DISABLE=1

#data=/home/will/data/dummy/cs21
data=/home/will/Downloads/dev_simu_linear_nonuniform_track1/
stage=1  # Modify this to control to start from which stage

dumpdir=data  # directory to put generated json file

# -- START Conv-TasNet Config
train_dir=$dumpdir/train
valid_dir=$dumpdir/dev
evaluate_dir=$dumpdir/eval
separate_dir=$dumpdir/eval
percentage=4
sample_rate=16000
segment=2  # seconds
cv_maxlen=3   # seconds
# Network config
N=256
L=60
B=256
H=384
P=3
X=8
R=4
norm_type=gLN
causal=1
mask_nonlinear='sigmoid'
C=1
# Training config
use_cuda=1
id=0,1,2,3
epochs=100
half_lr=1
early_stop=0
max_norm=3
# minibatch
shuffle=1
batch_size=16
num_workers=4
# optimizer
optimizer=adam
lr=3e-4
momentum=0
l2=0.01
# save and visualize
checkpoint=1
continue_from="exp/train_BIG/epoch51.pth.tar"
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
  preprocess_dev.py --in-dir $data --out-dir $dumpdir --sample-rate $sample_rate --corpus $corpus --percentage $percentage
fi


if [ -z ${tag} ]; then
  expdir=exp/train_r${sample_rate}_N${N}_L${L}_B${B}_H${H}_P${P}_X${X}_R${R}_C${C}_${norm_type}_causal${causal}_${mask_nonlinear}_epoch${epochs}_half${half_lr}_norm${max_norm}_bs${batch_size}_worker${num_workers}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}_`basename $train_dir`
else
  expdir=exp/train_${tag}
fi

mkdir $expdir

if [ $stage -le 3 ]; then
  echo "Stage 2: Evaluate separation performance"
  #${decode_cmd} --gpu ${ngpu} ${expdir}/evaluate.log \
    cs21_simu_eval.py \
    --model_path ${expdir}/final.pth.tar \
    --data_dir $valid_dir \
    --cal_sdr $cal_sdr \
    --use_cuda $ev_use_cuda \
    --sample_rate $sample_rate \
    --batch_size 1 \
    --multichannel $multichannel \
    #> $expdir/eval.log
fi
