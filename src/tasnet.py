import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import models, sdr


class Encoder(nn.Module):
    def __init__(self, N=512, L=32, stride=16, nchan=1):
        super(Encoder, self).__init__()
        self.N = N
        self.L = L
        self.stride = stride
        self.nchan = nchan
        self.encoder = nn.Conv1d(self.nchan, self.N, self.L, bias=False, stride=self.stride)
        self.layer_norm = nn.LayerNorm(self.N)
    
    def forward(self, input):
        return F.relu(self.layer_norm(self.encoder(input).view(input.shape[0], -1, self.N))) # B, N, L

class Decoder(nn.Module):
    def __init__(self, N=512, L=32, stride=16, nchan=1):
        super(Decoder, self).__init__()
        self.N = N
        self.L = L
        self.stride = stride
        self.nchan = nchan
        self.decoder = nn.ConvTranspose1d(self.N, self.nchan, self.L, bias=False, stride=self.stride)
    
    def forward(self, input):
        output = self.decoder(input) # B, N, L
        return output

class SeparationModule(nn.Module):
    def __init__(self, N=512, C=2, num_layers=2, repeats=2):
        super(SeparationModule, self).__init__()
        self.N=N
        self.C=C
        self.num_layers = num_layers
        self.repeats = repeats

        self.layer_norm = nn.LayerNorm(self.N)
        self.first_lstm = nn.LSTM(self.N,self.N*self.C,num_layers=self.num_layers)
        self.fully_connected = nn.Linear(self.N, self.N*self.C)
        self.lstm_blocks = nn.ModuleList([nn.LSTM(self.N*self.C,self.N*self.C,num_layers=self.num_layers)
                                             for i in range(self.repeats-1)])
        self.output_layer = nn.Linear(self.N*self.C,self.N*self.C)

    def forward(self, input):
        norm_weights = self.layer_norm(input)
        fc_layer = self.fully_connected(norm_weights) # expand dimension for residual
        current_block = self.first_lstm(norm_weights)[0] + fc_layer

        for lstm_block in self.lstm_blocks:
            current_block += lstm_block(current_block)[0]
        
        masks = torch.split(F.sigmoid(self.output_layer(current_block)),self.N,2)
        
        output = torch.Tensor()
        for mask in masks :  output=torch.cat((output,mask*input))
        #output = torch.Tensor([masks[i] * input for i in range(2)])
        # print(output, (output.shape))
        return output


# TasNet for Dereverberation
class TasNet(nn.Module):
    def __init__(self, N=512, sr=16000, L=32, X=2, R=2,
                 C=2, nchan=1, bidirectional=True):
        """
        N=N
        B=B (number of channels in bottleneck)
        sr=sampling rate
        L=
        """
        super(TasNet, self).__init__()

        # hyper parameters
        self.C = C

        self.N = N # number of filters
        
        #self.L = int(sr*L/1000)
        self.L = L  
        self.stride = self.L // 2
        self.R = R
        self.X=X
        self.sr = sr
        self.nchan = nchan

        self.bidirectional = bidirectional

        # input encoder
        self.encoder = Encoder(N=self.N, L=self.L, stride=self.stride, nchan=self.nchan)
        
        # separator
        self.separation = SeparationModule(N=self.N, C=self.C, num_layers=self.X, repeats=self.R)
    
        # output decoder
        self.decoder = Decoder(N=self.N, L=self.L, stride=self.stride, nchan=self.nchan)

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

        # waveform encoder
        enc_output = self.encoder(output) # B, N, L
        
        # generate masked weights
        masked_weights = torch.split(self.separation(enc_output.view(batch_size, -1, self.N)),batch_size,0)  # B, N, L
        
        # waveform decoder
        outputs = torch.Tensor()
        for mw in masked_weights: 
            output = self.decoder(mw.reshape(batch_size,self.N,-1)) # B, 1, L
            output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B, 1, L
            output = output.view(batch_size, 1, self.nchan, -1) 
            outputs = torch.cat((outputs,output), dim=1)

        return outputs # B, C, T

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(N=package['N'], sr=package['sr'], L=package['L'],
                    R=package['R'],
                    C=package['C'], causal=package['causal'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'N': model.N, 'L': model.L, 'B': model.B, 
             'R': model.R, 'C': model.C,
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
    nchan = 3
    x = torch.rand(13, nchan, 32000)
    nnet = TasNet(nchan=nchan)
    x = nnet(x)
    s1 = x[0]
    print(x.shape)


if __name__ == "__main__":
    test_conv_tasnet()
