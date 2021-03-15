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

data=/home/will/data/se/ConferencingSpeech2021/Train_dev_dataset/Evaluation_set/eval_data/task1
#data=/share/mini1/data/audvis/pub/se/mchan/mult/ConferencingSpeech/v1/eval_data/task1/
stage=2  # Modify this to control to start from which stage

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
batch_size=1
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

if [ -z ${tag} ]; then
  expdir=exp/train_r${sample_rate}_N${N}_L${L}_B${B}_H${H}_P${P}_X${X}_R${R}_C${C}_${norm_type}_causal${causal}_${mask_nonlinear}_epoch${epochs}_half${half_lr}_norm${max_norm}_bs${batch_size}_worker${num_workers}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}_`basename $train_dir`
else
  expdir=exp/train_${tag}
fi

if [ $stage -le 2 ]; then
  echo "Stage 4: Separate speech using Conv-TasNet"
  mkdir -p $expdir/{separate,originals}/task1/1/
  for data_type in real-recording semi-real-playback semi-real-realspk
  do
    echo "Separating $data_type"
    separate_out_dir=${expdir}/separate/task1/1/$data_type
    original_out_dir=${expdir}/originals/task1/1/$data_type
    #${decode_cmd} --gpu ${ngpu} ${separate_out_dir}/separate.log \
      separate.py \
      --model_path ${expdir}/final.pth.tar \
      --mix_json $separate_dir/$data_type.json \
      --out_dir ${separate_out_dir} \
      --originals_dir ${original_out_dir} \
      --use_cuda $ev_use_cuda \
      --sample_rate $sample_rate \
      --batch_size $batch_size \
      --num_workers $num_workers \
      --multichannel $multichannel \
      --figure $figures \
      --originals True \
      --append_file False \
      > $expdir/separate.log
  done
fi
