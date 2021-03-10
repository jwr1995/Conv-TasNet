#!/usr/bin/env python
# Created on 2018/12
# Author: Kaituo XU

import argparse
import json
import os
import random
from itertools import  compress

import soundfile as sf


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000, entries=None, complete=False):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    if not entries == None:
        wav_list = list(compress(wav_list,entries))
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples = sf.read(wav_path)[0]
        if complete==True:
            for channel in range(samples.shape[1]):
                 # file path, num samples, num channels, channel idx
                file_infos.append((wav_path, samples.shape[0],
                samples.shape[1], channel))
        else:
            file_infos.append((wav_path, samples.shape[0],
            samples.shape[1], 0))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)


def preprocess(args):
    for data_type in ['eval']:
        for source in ['real-recording', 'semi-real-playback','semi-real-realspk']:
            preprocess_one_dir(os.path.join(args.in_dir, source,1),
                               os.path.join(args.out_dir, data_type),
                               source,
                               sample_rate=args.sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Corpus data preprocessing")
    parser.add_argument('--in-dir', type=str, default=None,
                        help='Directory path of corpus including train, dev and eval')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Directory path to put output files')
    parser.add_argument('--sample-rate', type=int, default=8000,
                        help='Sample rate of audio file')
    parser.add_argument('--denoising', type=bool, default=True,
                        help='Set Conv-TasNet to perform denoising')
    parser.add_argument('--corpus', type=str, default="cs21",
                        help='Set corpus')
    parser.add_argument('--array', type=str, default='simu_non_uniform')
    parser.add_argument('--percentage', type=float, default=20.0)
    args = parser.parse_args()
    print(args)
    preprocess(args)
