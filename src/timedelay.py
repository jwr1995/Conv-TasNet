import torch.nn as nn
import cupy

def correlation(sig, ref):
    return cupy.correlate(cupy.array(sig),cupy.array(ref),'full')

class NormalizedTimeDelay(nn.Module):
    def __init__(self):
        pass

    def __correlation__(sig, ref):
        return cupy.correlate(cupy.array(sig),cupy.array(ref),'full')

    def forward(self, channels, idx)
        """
        C x L =  input
        idx = index
        """
        xcorrs = torch.Tensor((channels.shape[0],channels.shape[1]*2))
        for i, channel in enumerate(channels):
            xcorrs[i] = correlation(channel,channels[i])




correlation(a,b).device
