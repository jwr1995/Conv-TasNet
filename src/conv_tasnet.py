import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import models, sdr


# Conv-TasNet
class ConvTasNet(nn.Module):
    def __init__(self, N=512, B=128, sr=16000, L=40, X=8, R=3, H=512,
                 P=3, C=2, causal=False, num_channels=1, depth=1):
        """
        N=N
        B=B (number of channels in bottleneck)
        sr=sampling rate
        L=
        """
        super(ConvTasNet, self).__init__()

        # hyper parameters
        self.C = C

        self.N = N
        self.B = B
        self.H = H
        #self.L = int(sr*L/1000)
        self.L = L
        self.stride = self.L // 2

        self.X = X
        self.R = R
        self.P = P
        self.sr = sr

        self.num_channels = num_channels
        self.depth = depth

        self.causal = causal
        print("Causal:",self.causal)
        # input encoder
        if self.depth == 1:
            self.encoder = nn.Conv1d(self.num_channels, self.N, self.L, bias=False, stride=self.stride)
        elif self.depth == 2:
            self.encoder = nn.Sequential(nn.Conv1d(self.num_channels, self.N, self.L, bias=False, stride=self.stride),
                                nn.Conv1d(self.N, self.N, 1, bias=False, stride=1))
        else:
             self.encoder = nn.Conv1d(self.num_channels, self.N, self.L, bias=False, stride=self.stride)

        # TCN separator
        self.TCN = models.TCN(self.N, self.N*self.C, self.B, self.H,
                              self.X, self.R, self.P, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        if self.depth == 1:
            self.decoder = nn.ConvTranspose1d(self.N, self.num_channels, self.L, bias=False, stride=self.stride)
        elif self.depth == 2:
            self.decoder = nn.Sequential(nn.ConvTranspose1d(self.N,self.N, 1, bias=False, stride=1),
                                         nn.ConvTranspose1d(self.N, self.num_channels, self.L, bias=False, stride=self.stride))

        self.masks = None

    get_masks = lambda self : self.masks

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2,3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchan = input.size(1)
        nsample = input.size(2)
        rest = self.L - (self.stride + nsample % self.L) % self.L
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchan, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchan, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, input):

        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)

        #print(output.shape);exit()
        # waveform encoder
        enc_output = self.encoder(output)  # B, N, L

        #print(enc_output.shape,self.decoder(enc_output).shape)
        # generate masks
        self.masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.C, self.N, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * self.masks  # B, C, N, L


        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.C, self.N, -1))  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        if self.num_channels > 1:
            output = output.view(batch_size, self.C, self.num_channels, -1)  # B, C, T
        else:
            output = output.view(batch_size, self.C, -1)  # B, C, T

        return output

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(N=package['N'], B=package['B'], sr=package['sr'], L=package['L'],
                    X=package['X'], R=package['R'], H=package['H'], P=package['P'],
                    C=package['C'], causal=package['causal'], num_channels=package['num_channels'], depth=package['depth'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'N': model.N, 'L': model.L, 'B': model.B, 'H': model.H,
            'P': model.P, 'X': model.X, 'R': model.R, 'C': model.C,
            'causal': model.causal, 'sr': model.sr, 'num_channels': model.num_channels, 'depth': model.depth,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


def test_conv_tasnet():
    nchan=3
    x = torch.rand(13, nchan, 32000).cuda()
    nnet = ConvTasNet(C=1,num_channels=nchan,depth=1).cuda()
    #print(nnet.receptive_field)
    x = nnet(x)
    s1 = x[0]
    print(x.shape)


if __name__ == "__main__":
    test_conv_tasnet()
