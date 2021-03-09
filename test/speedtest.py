import time
import os, sys
import torch
#from torchsummary import summary
from pytorch_model_summary import summary

sys.path.append('/home/will/Projects/Conv-TasNet/src')
sys.path.append('/home/will/Projects/Conv-TasNet/ConferencingSpeech/baseline')

from multi_conv_tasnet import MultiConvTasNet
import utils

utils.IS_CUDA=False

N=256
L=60
B=256
H=384
P=3
X=8
R=4
norm_type='gLN'
causal=1
mask_nonlinear='sigmoid'
C=1
# Training config
use_cuda=1
id=0,1,2,3
epochs=10
half_lr=1
early_stop=0
max_norm=5
# minibatch
shuffle=1
batch_size=4
num_workers=0
# optimizer
optimizer='adam'
lr=1e-3
momentum=0
l2=0
sample_rate=16000
device = torch.device("cpu")
model = MultiConvTasNet(N, L, B, H, P, X, R,
                   C, norm_type=norm_type, causal=causal,
                   mask_nonlinear=mask_nonlinear)
model=model.eval()
model.to(device)
data = torch.rand([1,8,96000])
data=data.to(device)
tic=time.perf_counter()
output=model(data)
toc=time.perf_counter()
time=toc-tic
seg_time = 96000/sample_rate

print("Time taken:",str(time),"s")
print("RTF:",str(time/seg_time))

utils.IS_CUDA=True
model.cuda()

print(summary(model, data.cuda()))
