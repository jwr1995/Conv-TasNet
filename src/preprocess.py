#!/usr/bin/env python
# Created on 2018/12
# Author: Kaituo XU

import argparse
import json
import os

import librosa


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)


def preprocess(args):
    for data_type in ['tr', 'cv', 'tt']:
        for speaker in ['mix', 's1', 's2']:
            preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),
                               os.path.join(args.out_dir, data_type),
                               speaker,
                               sample_rate=args.sample_rate)

def preprocess_one_dir_chime3(in_dir, out_dir, out_filename, sample_rate=16000):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)

def preprocess_chime3(args):
    target_dict = {"mix":"isolated_6ch_track","s":"isolated_ext","v":"isolated_ext"}
    sets = [set for filepath in glob.glob(os.path.join(args.in_dir,"isolated_ext","*_simu"))]
    for set in sets:
        for key, value in target_dict:
            glob.glob()

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
    args = parser.parse_args()
    print(args)
    preprocess(args)
