import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool

T = 32000
M = 5
B = 200
L = 300
N = 256

### 1D ENCODER ###
print("### 1D Encoder Example ###")
mixture = torch.rand((B,T))
print(mixture.shape)
conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

#def forward_1d(mixture, channel):
mixture = torch.unsqueeze(mixture, 1)  # [B, 1, T]
print(mixture.shape)
mixture_w = F.relu(conv1d_U(mixture))  # [B, N, K]
print(mixture_w.shape)

### 2D ENCODER ###
print("\n### 2D Encoder Example ###")
mixtures = torch.rand((B,M,T),requires_grad=False)
print(mixture.shape)
conv2d_U = nn.Conv2d(1, N, kernel_size=L, stride=L // 2, bias=False,padding=math.ceil(L/2-M/2))

#def forward_1d(mixture, channel):
mixtures_u = torch.unsqueeze(mixtures, 1)  # [B, 1, M, T]
print(mixture.shape)
mixtures_w = F.relu(conv2d_U(mixtures_u)).squeeze()  # [B, N, K]

# mixture_w = F.relu(conv2d_U(mixture))  # [B, N, K]
# mixture_w = mixture_w.view(mixture_w.size(0), -1)
print(mixture_w.shape)

### MULTI 1D ENCODER ###
print("\n### Multi 1D Encoder Example")
def first_encode(mixtures):
    mixtures_w = torch.FloatTensor(B,M,N,T//(L//2)-1)
    print(mixtures_w.shape)
    for i in range(mixtures.shape[1]):
        mixtures_w[:,i]=F.relu(conv1d_U(torch.unsqueeze(mixtures[:,i,:],1)))

    return mixtures_w

def delay_encode(mixtures):
    weights=torch.FloatTensor(B,T//(L//2)-1)

mixtures_w = first_forward(mixtures)




print(mixtures_w.shape)
