#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse, glob, os

import torch

from data import AudioDataLoader, AudioDataset
from csdata import CSDataSet, DataConfiguration
from solver import Solver
from multitansnet import MultiTansNet, TansNet
from conv_tasnet import ConvTasNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
# Network architecture
parser.add_argument('--N', default=256, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=20, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=256, type=int,
                    help='Number of channels in bottleneck 1 x 1-conv block')
parser.add_argument('--H', default=256, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=4, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=2, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=1, type=int,
                    help='Number of speakers')
parser.add_argument('--norm_type', default='BN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')
# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='TasNet training',
                    help='Identifier for visdom run')
parser.add_argument('--datapath', default='/home/acp19jwr/mock/cs21', help='Path to tr and cv data')
#parser.add_argument('--datapath', default='/share/mini1/data/audvis/pub/se/mchan/mult/ConferencingSpeech/v1/ConferencingSpeech2021/simulation/data/wavs',
#                    help='Path to tr and cv data')
parser.add_argument('--channels', default='array',
                    help='Single channel = mono, single array = array or multiarray = multi')
parser.add_argument('--corpus', default='cs21',
                    help='cs21 for ConferencingSpeech 2021, wsj0  for wsj0-2mix')
def main(args):
    # Construct Solver
    # data
    if args.channels == 'array':
        # non_uniform configuration
        X_tr_wavs = glob.glob(os.path.join(args.datapath,'train','simu_non_uniform',"mix/*"))
        Y_tr_wavs = [X_tr_wav.replace("mix","nonreverb_ref") for X_tr_wav in X_tr_wavs]

        X_cv_wavs = glob.glob(os.path.join(args.datapath,'dev','simu_non_uniform',"mix/*"))
        Y_cv_wavs = [X_cv_wav.replace("mix","nonreverb_ref") for X_cv_wav in X_cv_wavs]

        tr_dataset = CSDataSet(X_tr_wavs, Y_tr_wavs, DataConfiguration.ARRAY)
        tr_loader = tr_dataset.loader(args.batch_size)

        cv_dataset = CSDataSet(X_cv_wavs, Y_cv_wavs, DataConfiguration.ARRAY)
        cv_loader = cv_dataset.loader(args.batch_size)
    elif args.channels == 'multi':
        pass

    # tr_dataset = AudioDataset(args.train_dir, args.batch_size,
    #                           sample_rate=args.sample_rate, segment=args.segment)
    # cv_dataset = AudioDataset(args.valid_dir, batch_size=1,  # 1 -> use less GPU memory to do cv
    #                           sample_rate=args.sample_rate,
    #                           segment=-1, cv_maxlen=args.cv_maxlen)  # -1 -> use full audio
    # tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
    #                             shuffle=args.shuffle,
    #                             num_workers=args.num_workers)
    # cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
    #                             num_workers=0)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    # model
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda==1 else "cpu")
    model = MultiTansNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                        norm_type=args.norm_type, causal=args.causal,
                       mask_nonlinear=args.mask_nonlinear)
   # model = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
    ##                   args.C, norm_type=args.norm_type, causal=args.causal,
        #               mask_nonlinear=args.mask_nonlinear)

    print(model)
    if torch.cuda.device_count()>1:
        model = model.to(torch.device('cuda:0')) 
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))
    
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
