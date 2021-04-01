import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utility import models, sdr


# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, N=512, B=128, sr=16000, L=32, X=8, R=3,
                 P=3, C=2, causal=False):
        """
        N=N
        B=B (number of channels in bottleneck)
        sr=sampling rate
        L=
        """
        super(TasNet, self).__init__()

        # hyper parameters
        self.C = C

        self.N = N
        self.B = B

        #self.L = int(sr*L/1000)
        self.L = L
        self.stride = self.L // 2

        self.B = X
        self.R = R
        self.P = P

        self.causal = causal

        # input encoder
        self.encoder = nn.Conv1d(1, self.N, self.L, bias=False, stride=self.stride)

        # TCN separator
        self.TCN = models.TCN(self.N, self.N*self.C, self.B, self.B*4,
                              self.X, self.R, self.P, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field

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
        enc_output = self.encoder(output)  # B, N, L

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.C, self.N, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L

        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.C, self.N, -1))  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.C, -1)  # B, C, T

        return output

def test_conv_tasnet():
    x = torch.rand(2, 32000)
    nnet = TasNet()
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()
