#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os

import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch

from data import AudioDataLoader, AudioDataset
from pit_criterion import cal_loss
from conv_tasnet import ConvTasNet
from multi_conv_tasnet import  MultiConvTasNet
from utils import remove_pad



parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=str, required=True,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--cal_sdr', type=int, default=0,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('--corpus', default="cs21", type=str)
parser.add_argument('--C', default=1, type=int)
parser.add_argument('--multichannel',default=False, type=bool)
parser.add_argument('--mix-label',default='mix',type=str)

def si_snr(estimated, original, eps=1e-8):
    # estimated = remove_dc(estimated)
    # original = remove_dc(original)
    target = pow_norm(estimated, original) * original / pow_np_norm(original)
    noise = estimated - target
    return 10 * np.log10((pow_np_norm(target) + eps) / (pow_np_norm(noise) + eps))

def pow_norm(s1, s2):
    return np.sum(s1 * s2)

def pow_np_norm(signal):
    """Compute 2 Norm"""
    return np.square(np.linalg.norm(signal, ord=2))

def evaluate(args):
    from pesq import pesq
    from pystoi.stoi import stoi
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load model
    if not args.multichannel:
        model = ConvTasNet.load_model(args.model_path)
    else:
        model = MultiConvTasNet.load_model(args.model_path)
    print(model)
    model.eval()
    #if args.use_cuda:
    if True:
        model.cuda()

    # Load data
    dataset = AudioDataset(args.data_dir, args.batch_size,
                           sample_rate=args.sample_rate, segment=-1, args=args,
                           mix_label=args.mix_label)

    data_loader = AudioDataLoader(multichannel=args.multichannel,dataset=dataset,
                                    batch_size=1, num_workers=2)
    results = []
    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            # Get batch data
            padded_mixture, mixture_lengths, padded_source = data # batch size x channels x samples

            #if args.use_cuda:
            if True:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            # Forward
            estimate_source = model(padded_mixture)  # [B, C, T]
            #print(estimate_source.shape)
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            # Remove padding and flat
            if args.multichannel:
                mixture = remove_pad(padded_mixture[:,0,:], mixture_lengths)
            else:
                mixture = remove_pad(padded_mixture, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)
            # NOTE: use reorder estimate source
            estimate_source = remove_pad(reorder_estimate_source,
                                         mixture_lengths)
            #print(mixture[0][0].shape,source[0].shape,estimate_source[0].shape)
            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):

                total_cnt += 1

                ref_sisdr = si_snr(mix, src_ref[0])
                enh_sisdr = si_snr(src_est[0], src_ref[0])
                ref_score = pesq(args.sample_rate,src_ref[0],mix, 'wb')#swap
                enh_score = pesq(args.sample_rate, src_ref[0],src_est[0], 'wb')#swap
                ref_stoi = stoi(src_ref[0], mix, args.sample_rate, extended=False)
                enh_stoi = stoi(src_ref[0], src_est[0], args.sample_rate, extended=False)
                ref_estoi = stoi(src_ref[0], mix, args.sample_rate, extended=True)
                enh_estoi = stoi(src_ref[0], src_est[0], args.sample_rate, extended=True)

                pesq_sum = 0
                stoi_sum = 0
                si_sdr = 0

                results.append([i,
                                {'pesq':[ref_score, enh_score],
                                 'stoi':[ref_stoi,enh_stoi],
                                 'si_sdr':[ref_sisdr, enh_sisdr],
                                }])
    filename = os.path.join(os.path.dirname(args.model_path),'results.csv')
    print(filename)
    with open(filename,'w') as wfid:
        wfid.writelines('ID,Ref PESQ,Est PESQ,Ref STOI,Est STOI,Ref SI-SDR,Est SI-SDR\n')

        for eval_score in results:
            utt_id, score = eval_score
            pesq = score['pesq']
            stoi = score['stoi']
            si_sdr = score['si_sdr']
            wfid.writelines(
                    'utt_{:s},{:.3f},{:.3f}, '.format(str(utt_id), pesq[0],pesq[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}, '.format(stoi[0],stoi[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}\n'.format(si_sdr[0],si_sdr[1])
                )

    #if args.cal_sdr:
    #    print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    #print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    C=(src_ref.shape[0])
    src_anchor = np.stack([mix]*C, axis=0)
    print(src_ref.shape, src_est.shape)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    if C==2:
        avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    elif C==1:
        avg_SDRi = (sdr[0]-sdr0[0])
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    C = src_ref.shape[0]
    if C ==2:
        sisnr1 = cal_SISNR(src_ref[0], src_est[0])
        sisnr2 = cal_SISNR(src_ref[1], src_est[1])
        sisnr1b = cal_SISNR(src_ref[0], mix)
        sisnr2b = cal_SISNR(src_ref[1], mix)
        # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
        #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
        # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
        avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    elif C==1:
        sisnr1 = cal_SISNR(src_ref[0], src_est[0])
        sisnr1b = cal_SISNR(src_ref[0], mix)
        avg_SISNRi = (sisnr1 - sisnr1b)

    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr




if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    evaluate(args)
