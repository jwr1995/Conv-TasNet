# Copied from ConferencingSpeech 2021
# https://github.com/ConferencingSpeech/ConferencingSpeech2021
# Copyright 2021-2022 Yukai Ju, Yihui Fu, Yanxin Hu
# Apache 2.0 Licence - see LICENCE

import torch
import torch as th
import torch.nn as nn

def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True)
    data = data - mean
    return data

def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2); #print(s1_s2_norm.shape)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_noise = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

def sisnr_rms_loss(s1, s2, eps=1e-8, l=0.75):
    sisnr=si_snr(s1, s2, eps=eps)
    rms_s1 = torch.sqrt(torch.mean(torch.square(s1)))
    rms_s2 = torch.sqrt(torch.mean(torch.square(s2)))
    return (-l*sisnr + (1.0-l)*torch.log10(torch.square(rms_s1-rms_s2)))


def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm

def sisnr_loss(inputs, labels):
    return -(si_snr(inputs, labels))

def test():
    x = th.Tensor([[[-1,-1]],[[-1,-1]]])
    y = th.Tensor([[[1,1]],[[1,1]]])
    print(x.shape)
    print(y.shape)
    z = sisnr_loss(x, y)
    print(z)
    print(z.item())

if __name__ == "__main__":
    test()
