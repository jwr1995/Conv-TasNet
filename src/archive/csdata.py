import glob, os
import soundfile as sf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum

class DataConfiguration(Enum):
    FLATTEN = 0
    MONO = 1
    ARRAY = 2


class CSDataSet(Dataset):
    def __init__(self, X_wavs, Y_wavs, config=DataConfiguration.MONO):
        if config == DataConfiguration.MONO or config == DataConfiguration.ARRAY:
            self.data_frame = pd.DataFrame(data=[[sf.read(X_wavs[i])[0].astype(np.float32).T,sf.read(Y_wavs[i])[0].astype(np.float32).T] for i in range(len(X_wavs))],columns=["X","Y"])
            self.config = config
            return
        if config == DataConfiguration.FLATTEN:
            data = []
            for i in range(len(X_wavs)):
                X_data = sf.read(X_wavs[i])[0].astype(np.float32).T
                y_data = sf.read(Y_wavs[i])[0].astype(np.float32).T
                for j in range(len(X_data)):
                    data.append([X_data[j],y_data[j]])
            self.data_frame = pd.DataFrame(data=data,columns=["X", "Y"])
            self.config=config
            return


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.config == DataConfiguration.MONO:
            sample = {'audio_in' : self.data_frame.iloc[idx,0][0][:], 'audio_out' : self.data_frame.iloc[idx,1][0][:]}
        elif self.config == DataConfiguration.FLATTEN:
            sample = {'audio_in' : self.data_frame.iloc[idx,0], 'audio_out' : self.data_frame.iloc[idx,1]}
        else:
            sample = {'audio_in' : self.data_frame.iloc[idx,0], 'audio_out' : self.data_frame.iloc[idx,1]}
        return torch.Tensor(sample['audio_in']), torch.Tensor(sample['audio_out'])

    def loader(self,size):
        return torch.utils.data.DataLoader(self,size)
