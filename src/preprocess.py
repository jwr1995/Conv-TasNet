#!/usr/bin/env python
# Created on 2018/12
# Author: Kaituo XU

import argparse
import json
import os
import glob
import random
import csv
from itertools import  compress
import numpy as np
from numpy import random
import soundfile as sf

from signalprocessing import rms

create_entries = lambda input_list, nfiles : list(random.permutation(np.array([True]*
                        nfiles+[False]*(len(input_list)-nfiles))))

def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000,
                        entries=None, complete=False, export_rms=False):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    try:
        wav_list = os.listdir(in_dir)

    except:
        wav_list = glob.glob(in_dir+"/*")

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
            file_infos.append((wav_path, samples.shape[0], samples.shape[1], 0))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)

    if export_rms:
        with open(os.path.join(out_dir,out_filename.split(".")[0]+'_rms.csv'),'w') as f:
            writer = csv.writer(f)
            for row in file_infos:
                writer.writerow([row[0],rms(sf.read(row[0])[0][0])])

def preprocess(args):
    if args.corpus == 'wsj0':
        for data_type in ['tr', 'cv', 'tt']:
            for speaker in ['mix', 's1', 's2']:
                preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),
                                   os.path.join(args.out_dir, data_type),
                                   speaker, sample_rate=args.sample_rate)
    elif args.corpus == 'cs21':
        for data_type in ['train', 'dev']:
            if data_type == 'train':
                flist=os.path.join(args.in_dir, args.array, 'noreverb_ref')
                entries=create_entries(os.listdir(flist), args.nfiles)
                for source in [args.mix_label, 'noreverb_ref']:
                    preprocess_one_dir(os.path.join(args.in_dir, args.array, source),
                                       os.path.join(args.out_dir, data_type),
                                       source,sample_rate=args.sample_rate,
                                       entries=entries)
            if data_type == 'dev':
                for source in [args.mix_label, 'noreverb_ref']:
                    preprocess_one_dir(os.path.join(args.dev_dir,
                                "simu_single_MA/dev_simu_linear_nonuniform_track1",
                                source),
                                os.path.join(args.out_dir, data_type),
                                source, sample_rate=args.sample_rate)

        for data_type in ['dev','eval']:
            for source in ['real-recording', 'semi-real-playback','semi-real-realspk']:
                if data_type == 'dev':
                    in_dir = os.path.join(args.dev_dir,source,'*/1')
                    export_rms=True
                else:
                    in_dir = os.path.join(args.eval_dir,source,'1')
                    export_rms=False
                preprocess_one_dir(in_dir,os.path.join(args.out_dir, data_type),
                                   source, sample_rate=args.sample_rate,export_rms=export_rms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Corpus data preprocessing")
    parser.add_argument('--in-dir', type=str, default=None,
                        help='Directory path of corpus including train, dev and eval')
    parser.add_argument('--dev-dir',type=str,default=None)#dev test
    parser.add_argument('--eval-dir',type=str,default=None)
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
    parser.add_argument('--mix-label', type=str, default="mix")
    parser.add_argument('--nfiles',type=int, default=6)
    args = parser.parse_args()
    print(args)
    preprocess(args)
