#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse

import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

from data import AudioDataLoader, AudioDataset
from solver import Solver
from conv_tasnet import ConvTasNet
from tasnet import TasNet
#from multi_conv_tasnet import MultiConvTasNet

torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
# General config
# Task related

bool_string = lambda the_string : True if the_string == 'True' else False
int_array_string = lambda the_string: [int(s) for s in the_string.split(',')]

parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=16000, type=int,
                    help='Sample rate')
parser.add_argument('--segments', default='4', type=int_array_string,
                    help='Segment length (seconds)')
parser.add_argument('--stops', default='', type=int_array_string,
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
parser.add_argument('--H', default=512, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=4, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=1, type=int,
                    help='Number of speakers')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=bool_string, default=False,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax','sigmoid'], help='non-linear to generate mask')
# parser.add_argument('--encoder_nonlinear',default='relu',type=str)
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
parser.add_argument('--batch_size', default=128, type=int,
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
parser.add_argument('--corpus',default='wsj0')
parser.add_argument('--array',default='simu_non_linear')
parser.add_argument('--multichannel',default=False, type=bool_string)
parser.add_argument('--mode', default="ss", type=str)
parser.add_argument('--subtract', default=False, type=bool_string)
parser.add_argument('--mix-label',default='mix',type=str)
parser.add_argument('--rms-dir',default=None,type=str)
parser.add_argument('--loss',default='sisnr',type=str)
parser.add_argument('--num_channels',default=1,type=int)
parser.add_argument('--depth',default=1,type=int, help="Encoder depth: 1 or 2")
parser.add_argument('--model',default="convtasnet",help="TasNet models available",choices=["convtasnet","tasnet"])

def main(args):
    # Construct Solver
    # data
    tr_dataset = AudioDataset(args.train_dir, args.batch_size, args=args,
                              sample_rate=args.sample_rate,
                              segments=args.segments, stops=args.stops, mode=args.mode,
                              mix_label=args.mix_label)
    cv_dataset = AudioDataset(args.valid_dir, batch_size=1, args=args,  # 1 -> use less GPU memory to do cv
                              sample_rate=args.sample_rate,
                              segments=[-1]*2, stops=[args.epochs], cv_maxlen=args.cv_maxlen,
                              mode=args.mode, mix_label=args.mix_label)  # -1 -> use full audio
    tr_loader = AudioDataLoader(multichannel=args.multichannel, subtract=args.subtract,
                                dataset=tr_dataset, batch_size=1,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
    cv_loader = AudioDataLoader(multichannel=args.multichannel, subtract=args.subtract,
                                dataset=cv_dataset, batch_size=1,
                                num_workers=0)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    #if args.multichannel == False:
    if args.model == "convtasnet":
        model = ConvTasNet(N=args.N, L=args.L, B=args.B, H=args.H, P=args.P, X=args.X, R=args.R,
                       C=args.C, causal=args.causal, depth=args.depth, num_channels=args.num_channels)
    elif args.model == "tasnet":
        model = TasNet(N=args.N, L=args.L, X=args.X, R=args.R, C=args.C, nchan=args.num_channels, bidirectional= not args.causal)
    # else:
    #     model = MultiConvTasNet(args.N,args.L, args.B, args.H, args.P, args.X, args.R,
    #                    args.C, norm_type=args.norm_type, causal=args.causal,
    #                    mask_nonlinear=args.mask_nonlinear)
    #print(model)
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
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
    print()
    print(args)
    main(args)
