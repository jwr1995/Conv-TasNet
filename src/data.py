# Created on 2018/12
# Author: Kaituo XU
"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.

Input:
    Mixtured WJS0 tr, cv and tt path
Output:
    One batch at a time.
    Each inputs's shape is B x T
    Each targets's shape is B x C x T
"""

import json
import math
import os
import glob
import csv 
import soundfile as sf

import numpy as np
import torch
import torch.utils.data as data

import librosa

from signalprocessing import rms


class AudioDataset(data.Dataset):

    def __init__(self, json_dir, batch_size, args=None, sample_rate=8000,
                segments=[1,2,4], stops=[10,20], cv_maxlen=8.0, multi=False, mode="ss", mix_label="mix", epochs=100):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioDataset, self).__init__()
        self.multi=multi
        self.mix_label=mix_label
        self.epochs=100
        self.segments=segments
        self.stops=stops
        self.batch_size=batch_size
        self.sample_rate=sample_rate
        self.segments=segments
        self.C = args.C
        self.mode=mode
        self.corpus=args.corpus
        self.cv_maxlen=cv_maxlen
        self.segment=segments[0]

        assert len(segments) == len(stops)+1, "There should be one more segment length than stopping epoch"

        self.segment_length = np.zeros(self.epochs,dtype=np.int8)
        last=0
        for i in range(len(self.stops)):
            self.segment_length[last:self.stops[i]] = self.segments[i]
            last = i
        self.segment_length[self.stops[i]:]

        if  args.corpus == "cs21":
            mix_json = os.path.join(json_dir, self.mix_label+'.json')
            s1_json = os.path.join(json_dir, 'noreverb_ref.json')
        elif args.corpus == "wsj0":
            mix_json = os.path.join(json_dir, self.mix_label+'.json')
            s1_json = os.path.join(json_dir, 's1.json')
            s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        if args.corpus == "wsj0":
            with open(s2_json, 'r') as f:
                s2_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        self.sorted_mix_infos = sort(mix_infos)
        self.sorted_s1_infos = sort(s1_infos)
        if args.C == 2 and mode == "ss":
            self.sorted_s2_infos = sort(s2_infos)

        self.minibatch = []
        self.update_segments(0,True)
        print("Mini batch length =",len(self.minibatch), mix_label)
        

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

    def update_segments(self,epoch,init=False):
        if (self.segment == self.segment_length[epoch]) and not init:
            return
        
        self.segment=self.segment_length[epoch]
        print(self.segment)
        if self.segment >= 0.0:
            # segment length and count dropped utts
            segment_len = int(self.segment * self.sample_rate)  # 4s * 8000/s = 32000 samples
            drop_utt, drop_len = 0, 0
            for cells in self.sorted_mix_infos:
                if len(cells) == 4:
                    _, sample, cs, c = cells
                elif len(cells) == 5:
                    _, sample, cs, c, _y = cells
                elif len(cells) == 6:
                    _, sample, cs, c, _y, _z = cells
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample
            print("Drop {} utts({:.2f} h) which is short than {} samples".format(
                drop_utt, drop_len/self.sample_rate/36000, segment_len))
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                num_segments = 0
                end = start
                if self.corpus == "cs21":
                    part_mix, part_s1 = [], []
                else:
                    part_mix, part_s1, part_s2 = [], [], []
                while num_segments < self.batch_size and end < len(self.sorted_mix_infos):
                    utt_len = int(self.sorted_mix_infos[end][1])
                    
                    if utt_len >= segment_len:  # skip too short utt
                        num_segments += math.ceil(utt_len / segment_len)
                        # Ensure num_segments is less than batch_size
                        print(num_segments,self.batch_size); 
                        # if num_segments > self.batch_size:
                        #     # if num_segments of 1st audio > batch_size, skip it
                        #     if start == end: end += 1
                        #     break
                        part_mix.append(self.sorted_mix_infos[end])
                        part_s1.append(self.sorted_s1_infos[end])
                        print((part_mix),(part_s1))
                        if self.C==2 and self.mode=="ss":
                            part_s2.append(self.sorted_s2_infos[end])   
                    end += 1
                if len(part_mix) > 0:
                    if self.corpus == "wsj0":
                        minibatch.append([part_mix, part_s1, part_s2,
                                      self.sample_rate, segment_len])
                    else:
                        minibatch.append([part_mix, part_s1,
                                      self.sample_rate, segment_len])
                if end == len(self.sorted_mix_infos):
                    break
                start = end
            
            self.minibatch = minibatch
        else:  # Load full utterance but not segment
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                end = min(len(self.sorted_mix_infos), start + self.batch_size)
                # Skip long audio to avoid out-of-memory issue
                if self.corpus == "wsj0" and int(self.sorted_mix_infos[start][1]) > self.cv_maxlen * self.sample_rate:
                    start = end
                    continue
                if self.corpus == "wsj0":
                    minibatch.append([self.sorted_mix_infos[start:end],
                                  self.sorted_s1_infos[start:end],
                                  self.sorted_s2_infos[start:end],
                                 self.sample_rate, self.segment])
                else:
                    minibatch.append([self.sorted_mix_infos[start:end],
                                  self.sorted_s1_infos[start:end],
                                  self.sample_rate, self.segment])
                if end == len(self.sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch
_MULTICHANNEL=False
_SUBTRACT=False
_USE_RMS=False
_RMS_MEAN=0.002
_RMS_VAR=0.001

def process_rms_csv(filename):
    with open(filename,'r') as f:
        reader=csv.reader(f)
        rms_table=[]
        for i, row in enumerate(reader):
            rms_table.append(float(row[1]))
        mean=np.mean(np.array(rms_table))
        variance=np.var(np.array(rms_table))
    return mean, variance

class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, multichannel=False, subtract=False, mix_label=None,
                rms_dir=None,*args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        global _MULTICHANNEL, _SUBTRACT, _RMS_MEAN, _RMS_VAR, _USE_RMS
        _MULTICHANNEL=multichannel
        _SUBTRACT=subtract
        if not rms_dir==None:
            _USE_RMS=True
            for filename in glob.glob(rms_dir+"/*.csv"):
                _RMS_MEAN, _RMS_VAR=process_rms_csv(filename)
        self.collate_fn = _collate_fn

def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    global _MULTICHANNEL, _SUBTRACT
    mixtures, sources = load_mixtures_and_sources(batch[0],
                        multichannel=_MULTICHANNEL,
                        subtract=_SUBTRACT)

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[bool(_MULTICHANNEL)] for mix in mixtures])
    # perform padding and convert to tensor
    pad_value = 0

    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)

    # N x T x C -> N x C x T
    if _SUBTRACT:
        sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    else:
        sources_pad = sources_pad.reshape((sources_pad.shape[0], 1,
                    sources_pad.shape[1])).contiguous()

    return mixtures_pad, ilens, sources_pad


# Eval data part
from preprocess import preprocess_one_dir


class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000, mix_label="mix"):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, mix_label,
                               sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, mix_label+'.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        self.sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(self.sorted_mix_infos), start + batch_size)
            minibatch.append([self.sorted_mix_infos[start:end],
                              sample_rate])
            if end == len(self.sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """
    def __init__(self, multichannel=False, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        global _MULTICHANNEL
        _MULTICHANNEL = multichannel
        self.collate_fn = _collate_fn_eval

def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    global _MULTICHANNEL
    # batch should be located in list
    assert len(batch) == 1
    if _MULTICHANNEL:
        mixtures, filenames, channels = load_mixtures(batch[0])
    else:
        mixtures, filenames, channels = load_mixtures(batch[0])
    # get batch of lengths of input sequences
    if bool(_MULTICHANNEL):
        ilens = np.array([mix.shape[int(_MULTICHANNEL)] for mix in mixtures])
    else:
        ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor 
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value, multichannel=_MULTICHANNEL)
    ilens = torch.from_numpy(ilens)
    if channels:
        return mixtures_pad, ilens, filenames, channels
    else:
        return mixtures_pad, ilens, filenames, channels

def remove_dc(data):
    mean = np.mean(data, -1)
    data = data - mean
    return data

rms_normalization = lambda wav, target_loudness=100, ref_loudness=100 : (10**(target_loudness/20))*(10**(-ref_loudness/20))/np.sqrt(np.mean(np.square(wav)))*wav

def peak_normalize(sig):
    max_val = np.max(np.abs(sig))
    return sig*(1/max_val)

# ------------------------------ utils ------------------------------------
def load_mixtures_and_sources(batch, multichannel=False, subtract=False, eval=False):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """

    mixtures, sources = [], []
    if len(batch) == 5:
        C=2
        mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch
        zipped = zip(mix_infos, s1_infos, s2_infos)
    # elif len(batch) == 5 and _NUM_SPEAKERS == 1:
    else:
        C=1
        mix_infos, s1_infos, sample_rate, segment_len = batch
        zipped = zip(mix_infos, s1_infos)
    # for each utterance

    for packet in zipped:
        if C==2 and not subtract:
            mix_info, s1_info, s2_info = packet
        else:
            mix_info, s1_info = packet
            channel=mix_info[-3]
            start_sample=s1_info[-2]
            end_sample=mix_info[-1]

        mix_path = mix_info[0]
        s1_path = s1_info[0]
        assert mix_info[1] == s1_info[1]
        if C==2 and not subtract:
            s2_path = s2_info[0]
            assert s1_info[1] == s2_info[1]
        # read wav file
        #mix, _ = librosa.load(mix_path, sr=sample_rate)
        #s1, _ = librosa.load(s1_path, sr=sample_rate)
        if multichannel == False:
            # 1 x samples
            mix = remove_dc(sf.read(mix_path)[0][start_sample:end_sample,channel])
            if eval:
                s1 = sf.read(s1_path)[0][:,channel]
            else:
                #s1 = rms_normalization(remove_dc(sf.read(s1_path)[0][start_sample:end_sample,channel]),target_loudness=75)
                s1 = sf.read(s1_path)[0][start_sample:end_sample,channel]
        else:
            # channels x samples
            mix = sf.read(mix_path)[0].T
            if eval:
                #s1 = rms_normalization(remove_dc(sf.read(s1_path)[0][:,0]),target_loudness=75)
                s1 = sf.read(s1_path)[0][:,0]
            else:
                #s1 = rms_normalization(remove_dc(sf.read(s1_path)[0][:,0]),target_loudness=75)
                s1 = sf.read(s1_path)[0][:,0]
            # import matplotlib.pyplot as plt
            # plt.plot(s1)
            # plt.show()

        if C==2:
            s2 = sf.read(s2_path)[:,0]
            #s2 = rms_normalization(sf.read(s2_path)[:,0],target_loudness=70)
        elif subtract:
            s2 = mix-s1

        # global _USE_RMS
        # if _USE_RMS:
        #     global _RMS_MEAN, _RMS_VAR
        #     mix = rms_normalize(mix,np.random.normal(_RMS_MEAN,_RMS_VAR))

        # merge s1 and s2
        if C==2 or subtract:
            s = np.dstack((s1, s2))[0]  # T x C, C = 2
        else:
            s=s1
            #s = np.dstack((s1))[0]
        utt_len = mix.shape[-1]
        if segment_len >= 0:
            # segmentAudioDataset
            for i in range(0, utt_len - segment_len + 1, segment_len):
                if not multichannel:
                    mixtures.append(mix[i:i+segment_len])
                else:
                    mixtures.append(mix[:,i:i+segment_len])
                sources.append(s[i:i+segment_len])
            if utt_len % segment_len != 0:
                if not multichannel:
                    mixtures.append(mix[-segment_len:])
                else:
                    mixtures.append(mix[:,-segment_len:])
                sources.append(s[-segment_len:])
        else:  # full utterance
            mixtures.append(mix)
            sources.append(s)
    return mixtures, sources

def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    global _MULTICHANNEL
    mixtures, filenames, channels = [], [], []
    mix_infos, sample_rate = batch
    if (len(mix_infos[0]))==6: C=1
    else: C=None

    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file

        if C == 1:
            if not bool(_MULTICHANNEL):
                channel = mix_info[-3]
                mix = sf.read(mix_path)[0].T[channel]
            else:
                channel = 0
                mix = sf.read(mix_path)[0].T
            channels.append(channel)
        else:
            mix, _ = librosa.load(mix_path, sr=sample_rate)

        mixtures.append(mix)
        filenames.append(mix_path)

    return mixtures, filenames, channels

def pad_list(xs, pad_value,multichannel=False):

    n_batch = len(xs)

    if multichannel:
        pos = 1
    else:
        pos = 0

    max_len = max(x.size(pos) for x in xs)

    if multichannel:
        pad = xs[0].new(n_batch, xs[0].size()[0], max_len).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :, :xs[i].size(1)] = xs[i]
    else:
        pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    import sys
    json_dir, batch_size = "exp/temp/mix.json",3
    dataset = AudioDataset(json_dir, int(batch_size))
    data_loader = AudioDataLoader(dataset, batch_size=1,
                                  num_workers=4)
    for i, batch in enumerate(data_loader):
        mixtures, lens, sources = batch
        print(i)
        print(mixtures.size())
        print(sources.size())
        print(lens)
        if i < 10:
            print(mixtures)
            print(sources)
