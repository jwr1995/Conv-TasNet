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
elif [[ $(hostname) = Aithon ]]
then
  conda activate base
  echo "Using environment: base"
else
  conda activate cs21
  echo "Using environment: cs21"
fi

#export LDLIBRARYPATH=/share/mini1/sw/std/cuda/
#_LIBRARY_PATH=${LD_LIBRARY_PATH}:/share/mini1/sw/std/cuda/cuda10.1/x86_64/lib64/:/share/mini1/sw/std/cuda/cuda10.1/x86_64/include/:/share/mini1/sw/std/cuda/cuda10.1/cuda/:/share/mini1/sw/std/cuda/cuda10.1/x86_64/lib64/stubs

export NCCL_SOCKET_IFNAME=virbr0
export NCCL_IB_DISABLE=1

if [[ $(hostname) = Aithon ]]
then
  train_data=/home/will/data/dummy/cs21/train
  dev_data=/home/will/data/se/ConferencingSpeech2021/Train_dev_dataset/Development_test_set/
  eval_data=/home/will/data/se/ConferencingSpeech2021/Train_dev_dataset/Evaluation_set/eval_data/task1
else
  data=/share/mini1/data/audvis/pub/se/mchan/mult/ConferencingSpeech/v1/ConferencingSpeech2021/simulation/data/wavs
  eval_data=/share/mini1/data/audvis/pub/se/mchan/mult/ConferencingSpeech/v1/eval_data/task1/
fi

stage=4  # Modify this to control to start from which stage

dumpdir=data  # directory to put generated json file

# -- START Conv-TasNet Config
train_dir=$dumpdir/train
valid_dir=$dumpdir/dev
evaluate_dir=$dumpdir/eval
separate_dir=$dumpdir/eval
percentage=100
sample_rate=16000
segment=2  # seconds
cv_maxlen=3   # seconds
# Network config
if [[ $(hostname) = Aithon ]]
then
  N=256
  L=60
  B=128
  H=256
  P=3
  X=8
  R=4
else
  N=256
  L=60
  B=256
  H=512
  P=3
  X=8
  R=4
fi
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
if [[ $(hostname) = Aithon ]]
then
  batch_size=4
  num_workers=4
else
  batch_size=16
  num_workers=8
fi
# optimizer
optimizer=adam
lr=3e-4
momentum=0
l2=0.01
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
mix_label="reverb_ref"

# -- END Conv-TasNet Config

# exp tag
tag="BIG" # tag for managing experiments.

ngpu=1  # always 1

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh


if [ $stage -le 1 ]; then
  echo "Stage 1: Generating train & dev json files including wav path and duration"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  preprocess.py \
  --in-dir $train_data \
  --dev-dir $dev_data \
  --eval-dir $eval_data \
  --out-dir $dumpdir \
  --sample-rate $sample_rate \
  --corpus $corpus \
  --percentage $percentage \
  --mix-label $mix_label
fi

if [ -z ${tag} ]; then
  expdir=exp/train_r${sample_rate}_N${N}_L${L}_B${B}_H${H}_P${P}_X${X}_R${R}_C${C}_${norm_type}_causal${causal}_${mask_nonlinear}_epoch${epochs}_half${half_lr}_norm${max_norm}_bs${batch_size}_worker${num_workers}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}_`basename $train_dir`
else
  expdir=exp/train_${tag}
fi

mkdir -p $expdir
cp run.sh $expdir/run.sh

if [ $stage -le 2 ]; then
  echo "Stage 2: Training"
  #${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    train.py \
    --train_dir $train_dir \
    --valid_dir $valid_dir \
    --sample_rate $sample_rate \
    --segment $segment \
    --cv_maxlen $cv_maxlen \
    --N $N \
    --L $L \
    --B $B \
    --H $H \
    --P $P \
    --X $X \
    --R $R \
    --C $C \
    --norm_type $norm_type \
    --causal $causal \
    --mask_nonlinear $mask_nonlinear \
    --use_cuda $use_cuda \
    --epochs $epochs \
    --half_lr $half_lr \
    --early_stop $early_stop \
    --max_norm $max_norm \
    --shuffle $shuffle \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --optimizer $optimizer \
    --lr $lr \
    --momentum $momentum \
    --l2 $l2 \
    --save_folder ${expdir} \
    --checkpoint $checkpoint \
    --continue_from "$continue_from" \
    --print_freq ${print_freq} \
    --visdom $visdom \
    --visdom_epoch $visdom_epoch \
    --visdom_id "$visdom_id"\
    --corpus $corpus \
    --array $array \
    --multichannel $multichannel \
    --mix-label $mix_label \
    #>> $expdir/train.log
fi

cp run.sh.log $expdir/run.sh.log

# if [ $stage -le 3 ]; then
#   echo "Stage 3: Evaluate separation performance"
#   #${decode_cmd} --gpu ${ngpu} ${expdir}/evaluate.log \
#     evaluate.py \
#     --model_path ${expdir}/final.pth.tar \
#     --data_dir $evaluate_dir \
#     --cal_sdr $cal_sdr \
#     --use_cuda $ev_use_cuda \
#     --sample_rate $sample_rate \
#     --batch_size $batch_size \
#     --multichannel $multichannel \
#     > $expdir/eval.log
# fi

if [ $stage -le 3 ]; then
  echo "Stage 3: Evaluate separation performance"
  #${decode_cmd} --gpu ${ngpu} ${expdir}/evaluate.log \
    evaluate_se.py \
    --model_path ${expdir}/final.pth.tar \
    --data_dir $valid_dir \
    --cal_sdr $cal_sdr \
    --use_cuda $ev_use_cuda \
    --sample_rate $sample_rate \
    --batch_size 1 \
    --multichannel $multichannel \
    --mix-label $mix_label
    #> $expdir/eval.log
fi

if [ $stage -le 4 ]; then
  echo "Stage 4: Separate speech using Conv-TasNet"
  mkdir -p $expdir/{separate,originals}
  mkdir -p $expdir/{separate,originals}/task1
  mkdir -p $expdir/{separate,originals}/task1/1/
  for fname in $(dir $separate_dir)
  do
    echo "Separating $fname"
    separate_out_dir=${expdir}/separate/task1/1/$(basename -s .json fname)
    original_out_dir=${expdir}/originals/task1/1/$(basename -s .json fname)
  #${decode_cmd} --gpu ${ngpu} ${separate_out_dir}/separate.log \
    separate.py \
    --model_path ${expdir}/final.pth.tar \
    --mix_json $separate_dir/$fname \
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
    #> $expdir/separate.log
  done
fi
