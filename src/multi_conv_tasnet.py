# Created on 2018/12
# Author: Kaituo XU

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool
import sys,math
sys.path.append("/home/will/Projects/Conv-TasNet/src")
import utils
from conv_tasnet import Encoder, Decoder, TemporalConvNet

EPS = 1e-8

class MultiConvTasNet(nn.Module):
    def __init__(self, N, L, B, H, P, X, R, C, norm_type="gLN", causal=False,
                 mask_nonlinear='relu', n_channels=8):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(MultiConvTasNet, self).__init__()
        # Hyper-parameter
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        # Components
        self.encoder = Encoder(L, N)
        self.spatial_encoder = SpatialEncoder(L,N,n_channels)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal,
                        mask_nonlinear,multiplier=2)
        #self.separator2 = TemporalConvNet(N, B, H, P, X, R2, C, norm_type, causal, mask_nonlinear)
        self.decoder = Decoder(N, L)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, C, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture[:,0,:])
        spatial_mixture = self.spatial_encoder(mixture)
        est_mask = self.separator(torch.cat((mixture_w,spatial_mixture),dim=1))
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['N'], package['L'], package['B'], package['H'],
                    package['P'], package['X'], package['R'], package['C'],
                    norm_type=package['norm_type'], causal=package['causal'],
                    mask_nonlinear=package['mask_nonlinear'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'N': model.N, 'L': model.L, 'B': model.B, 'H': model.H,
            'P': model.P, 'X': model.X, 'R': model.R, 'C': model.C,
            'norm_type': model.norm_type, 'causal': model.causal,
            'mask_nonlinear': model.mask_nonlinear,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

class SpatialEncoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, L, N, n_channels=8):
        super(SpatialEncoder, self).__init__()

        # Hyper-parameter
        self.L, self.N, self.n_channels = L, N, n_channels
        # Components
        # 50% overlap
        #self.conv2d_U = nn.Conv2d(1, N, kernel_size=L, stride=L // 2, bias=False,padding=(int(math.ceil(L/2)),0))
        self.encoders=nn.ModuleList([Encoder(L,int(N))]*n_channels)
        self.lstm = nn.LSTM(N*n_channels,N,2,batch_first=True,bias=False)
        self.h = None
        self.c = None

    def forward(self, mixtures):
        """
        Args:
            mixture: [M, n_channels, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        self.lstm.flatten_parameters()
        encoded_mixtures =[F.relu(e(mixtures[:,i,:]))
                    for i, e in enumerate(self.encoders)]
        mixtures_s = torch.stack(encoded_mixtures)
        C, M, N, K = mixtures_s.shape
        mixtures_l = mixtures_s.reshape((M,K,C*N))

        mixture_w, (self.h, self.c) = self.lstm(mixtures_l)
        mixture_o = F.relu(mixture_w)
        return mixture_o.movedim(1,2)


if __name__ == "__main__":
    utils.IS_CUDA=False
    L= 50
    N=256
    B=5
    n_channels=8
    fs=16000
    data = torch.rand((B,n_channels,6*fs))
    encoder2D = Encoder2D(L, N)
    out = encoder2D(data)
    print(out.shape)

    mono_data = torch.rand((B,6*fs))
    encoder = Encoder(L, N)
    out= encoder(mono_data)
    print(out.shape)

    model = MultiConvTasNet(N, L, B, 3, 3, 3, 3, 1, norm_type="gLN", causal=False,
                 mask_nonlinear='relu')
    out = model(data)
    print(out.shape)

#     torch.manual_seed(123)
#     M, N, L, T = 2, 3, 4, 12
#     K = 2*T//L-1
#     B, H, P, X, R, C, norm_type, causal = 2, 3, 3, 3, 2, 2, "gLN", False
#     mixture = torch.randint(3, (M, T))
#     # test Encoder
#     encoder = Encoder(L, N)
#     encoder.conv1d_U.weight.data = torch.randint(2, encoder.conv1d_U.weight.size())
#     mixture_w = encoder(mixture)
#     print('mixture', mixture)
#     print('U', encoder.conv1d_U.weight)
#     print('mixture_w', mixture_w)
#     print('mixture_w size', mixture_w.size())
#
#     # test TemporalConvNet
#     separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type=norm_type, causal=causal)
#     est_mask = separator(mixture_w)
#     print('est_mask', est_mask)
#
#     # test Decoder
#     decoder = Decoder(N, L)
#     est_mask = torch.randint(2, (B, K, C, N))
#     est_source = decoder(mixture_w, est_mask)
#     print('est_source', est_source)
#
#     # test Conv-TasNet
#     conv_tasnet = ConvTasNet(N, L, B, H, P, X, R, C, norm_type=norm_type)
#     est_source = conv_tasnet(mixture)
#     print('est_source', est_source)
#     print('est_source size', est_source.size())
