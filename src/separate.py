#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os

import librosa
import soundfile as sf
import torch

from data import EvalDataLoader, EvalDataset
from conv_tasnet import ConvTasNet
#from multi_conv_tasnet import MultiConvTasNet
from utils import remove_pad

bool_string = lambda the_string : True if the_string == 'True' else False

parser = argparse.ArgumentParser('Separate speech using Conv-TasNet')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
parser.add_argument('--mix_dir', type=str, default=None,
                    help='Directory including mixture wav files')
parser.add_argument('--mix_json', type=str, default=None,
                    help='Json file including mixture wav files')
parser.add_argument('--out_dir', type=str, default='exp/result',
                    help='Directory putting separated wav files')
parser.add_argument('--originals_dir', type=str, default=None,
                    help='Directory putting original wav files')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU to separate speech')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('--multichannel',default=False, type=bool_string)
parser.add_argument('--num_workers',default=0,type=int)
parser.add_argument('--figures',default=False,type=bool)
parser.add_argument('--originals',default=True,type=bool)
parser.add_argument('--append_file',default=False,type=bool)
parser.add_argument('--mix-label',default="mix",type=str)


def separate(args):
    if args.mix_dir is None and args.mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "mix_json is ignored.")
    #print(args.multichannel); exit()
    # Load model
    # if args.multichannel:
    #     model = MultiConvTasNet.load_model(args.model_path)
    # else:
    model = ConvTasNet.load_model(args.model_path)
    print(model)
    model.figures = args.figures
    model.eval()
    if True:
    #if args.use_cuda:
        model.cuda()

    # Load data
    eval_dataset = EvalDataset(args.mix_dir, args.mix_json,
                               batch_size=args.batch_size,
                               sample_rate=args.sample_rate)
    eval_loader =  EvalDataLoader(multichannel=args.multichannel,
                                dataset=eval_dataset, batch_size=1,
                                num_workers=args.num_workers)
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.originals_dir == None:
        os.makedirs(args.originals_dir, exist_ok=True)

    def write(inputs, filename, sr=args.sample_rate):
        #librosa.output.write_wav(filename, inputs, sr)# norm=True)
        sf.write(filename, inputs, sr, 'PCM_16')

    if args.figures: import matplotlib.pyplot as plt

    with torch.no_grad():
        for (i, data) in enumerate(eval_loader):
            # Get batch data
            mixture, mix_lengths, filenames, channels = data
            #if args.use_cuda:
            if True:
                mixture, mix_lengths = mixture.cuda(), mix_lengths.cuda()
            # Forward
            estimate_source = model(mixture)  # [B, C, T];
            est_masks = model.get_masks()
            # Remove padding and flat
            flat_estimate = remove_pad(estimate_source, mix_lengths)

            mixture = remove_pad(mixture, mix_lengths, multichannel=args.multichannel)
            # Write result
            for i, filename in enumerate(filenames):

                if args.originals:
                    if args.originals_dir == None:
                        filename = os.path.join(args.out_dir,
                                            os.path.basename(filename).strip('.wav'))
                    else:
                        filename = os.path.join(args.originals_dir,
                                            os.path.basename(filename).strip('.wav'))
                    if args.multichannel:
                        write((mixture[i][0]), filename + '.wav',args.sample_rate)
                    else:
                        write(mixture[i], filename + '.wav',args.sample_rate)
                C = flat_estimate[i].shape[0]
                for c in range(C):
                    filename = os.path.join(args.out_dir,
                                        os.path.basename(filename).strip('.wav'))
                    if args.append_file:
                        write((flat_estimate[i][c]), filename +".CH"+str(channels[i]) + '_s{}.wav'.format(c+1),args.sample_rate)
                    else:
                        write((flat_estimate[i][c]), filename + '.wav')

                fig = plt.figure(figsize=[8,10])

                ax1 = fig.add_subplot(311)
                if not args.multichannel:
                    pxx,  freq, t, cax =ax1.specgram((mixture[i]),
                    Fs=args.sample_rate,scale='dB',NFFT=256, mode='magnitude',
                    vmin=-100)
                else:
                    pxx,  freq, t, cax =ax1.specgram((mixture[i][0]),
                    Fs=args.sample_rate,NFFT=256,scale='dB', mode='magnitude',
                    vmin=-100)
                ax1.title.set_text('Mixture')
                fig.colorbar(cax,ax=ax1).set_label("Loudness [dB]")

                import numpy as np
                ax2 = fig.add_subplot(312)
                mat = ax2.imshow(10*np.log(est_masks[i][0].cpu()), cmap='viridis',
                 aspect="auto",vmin=-100)
                ax2.title.set_text('Masks')
                plt.gca().xaxis.tick_bottom()
                fig.colorbar(mat,ax=ax2).set_label("Magnitude [dB]")

                ax3 = fig.add_subplot(313)
                pxx,  freq, t, cax3 =ax3.specgram(flat_estimate[i][0],
                                    Fs=args.sample_rate,NFFT=256, scale='dB',
                                    vmin=-100)
                ax3.title.set_text('Estimated Source')
                #plt.colorbar(cmap='binary')
                fig.colorbar(cax3,ax=ax3).set_label("Loudness [dB]")
                #plt.show()

                fig.savefig(filename+ '.png')
                plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    separate(args)
