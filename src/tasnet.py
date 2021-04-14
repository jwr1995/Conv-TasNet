import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import models, sdr


# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, N=512, sr=16000, L=32, R=3, H=512,
                 C=2, bidirectional=False):
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
        self.H = H
        #self.L = int(sr*L/1000)
        self.L = L
        self.stride = self.L // 2

        self.R = R
        self.sr = sr

        self.bidirectional = bidirectional

        # input encoder
        self.encoder_U = nn.Conv1d(1, self.N, self.L, bias=False, stride=self.stride)
        self.encoder_V = nn.Conv1d(1, self.N, self.L, bias=False, stride=self.stride)
        self.layer_norm = nn.LayerNorm(self.N)
        # TCN separator
        self.lstm = nn.LSTM(self.N,self.H,self.R)
        self.softmax = nn.Softmax(self.N*self.C)

        # output decoder
        self.decoder = nn.ConvTranspose1d(self.N, 1, self.L, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.L - (self.stride + nsample % self.L) % self.L
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, input):

        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)

        # waveform encoder
        enc_output = F.relu(self.encoder_U(output)) * F.sigmoid(self.encoder_V(output)) # B, N, L

        # generate masks
        masks = self.softmax(self.layer_norm(enc_output)).view(batch_size, self.C, self.N, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L

        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.C, self.N, -1))  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
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
                    C=package['C'], causal=package['causal'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'N': model.N, 'L': model.L, 'B': model.B, 'H': model.H,
            'P': model.P, 'X': model.X, 'R': model.R, 'C': model.C,
            'causal': model.causal, 'sr': model.sr,
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
    x = torch.rand(2, 32000).cuda()
    nnet = ConvTasNet().cuda()
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()
