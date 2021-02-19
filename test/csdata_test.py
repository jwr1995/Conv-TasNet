import glob, os, sys
import torch
import torch.optim
import torch.nn as nn
sys.path.append('/home/will/Projects/Conv-TasNet/src')
from csdata import CSDataSet, DataConfiguration
from multitansnet import MultiTansNet, TansNet
import pit_criterion

data_root="/home/will/data/dummy/cs21"
train_data=os.path.join(data_root,"train")
train_uniform_data=os.path.join(train_data,"simu_non_uniform")
X_wavs = glob.glob(os.path.join(train_uniform_data,"mix/*"))
Y_wavs = glob.glob(os.path.join(train_uniform_data,"nonreverb_ref/*"))
Z_wavs = glob.glob(os.path.join(train_uniform_data,"reverb_ref/*"))

train_set = CSDataSet(X_wavs, Y_wavs, config=DataConfiguration.ARRAY)
#print(len(train_set))

N=256
L=50
B=20
H=32
P=2
X=3
R=4
C=1

torch.cuda.current_device()
tansnet = TansNet(N, L, B, H, P, X, R, norm_type="gLN", causal=True,
 mask_nonlinear='relu').cuda()
mtansnet = MultiTansNet(N, L, B, H, P, X, R, norm_type="gLN", causal=True,
        mask_nonlinear='relu').cuda()

tansnet.cuda()
mtansnet.cuda()

train_generator = train_set.loader(10)

optimizer = torch.optim.SGD(mtansnet.parameters(),
                             lr=0.01,
                             momentum=0.9)
optimizer.zero_grad()
loss_fn =nn.MSELoss()
for batch in train_generator:
    optimizer.zero_grad()
    print(batch[0].shape)
    Y_hat=mtansnet(torch.Tensor(batch[0]).cuda())
    Y = torch.reshape(batch[1],(batch[1].shape[0]*batch[1].shape[1],batch[1].shape[2])).unsqueeze(1).cuda()
    print(Y.shape,Y_hat.shape)
    loss=loss_fn(Y, Y_hat)
    print(loss.item())
    loss.backward()
    optimizer.step()
