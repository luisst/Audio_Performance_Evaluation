#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:50:31 2022

@author: luis
"""

import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram as MelSpec

def normalize_0_to_1(matrix):
    """Normalize matrix to a 0-to-1 range; code from
    'how-to-efficiently-normalize-a-batch-of-tensor-to-0-1' """
    d1, d2, d3 = matrix.size() #original dimensions
    matrix = matrix.reshape(d1, -1)
    matrix -= matrix.min(1, keepdim=True)[0]
    matrix /= matrix.max(1, keepdim=True)[0]
    matrix = matrix.reshape(d1, d2, d3)
    return matrix

npy_output_path = '/home/luis/Documents/TS_EN_phrases/slim_feb05_ALL/eng_feb05_slim_gt_ext_x4.npy'
old_transcr_path = '/home/luis/Documents/TS_EN_phrases/slim_feb05_ALL/transcript.txt'
dst_dir_path = './'


SR = 16000
mels = 128
sr_coeff = SR / 1000 #divide by 1000 to save audio's duration in milisecs

np_array = np.load(npy_output_path, allow_pickle=True)

#Grab audio waves from {npy_file} and transcript lines from {transcript}
f = open(old_transcr_path, 'r')
lines = f.readlines()
f.close()

idx = 78
line = lines[idx]

audio_samples_np = np_array[idx]

#Samples from npy file are zero padded; these 3 lines will 'unpad'.
audio_orig_path, text = line.strip().split('\t')
# orig_samples, _ = torchaudio.load(audio_orig_path)
audio_dur = int(len(audio_samples_np) / sr_coeff) # audio's duration

#Convert from numpy array of integers to pytorch tensor of floats
audio_samples_pt = torch.from_numpy(audio_samples_np)

# temp_path = f"/home/luis/Desktop/{idx}.wav"
# torchaudio.save(temp_path, audio_samples_pt, SR)

audio_samples_pt = torch.unsqueeze(audio_samples_pt, 0)
#Uncomment line below if npy output is int
audio_samples_pt = audio_samples_pt.type(torch.FloatTensor)

#Calculate spectrogram and normalize
spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(audio_samples_pt)
spctrgrm = normalize_0_to_1(spctrgrm)
spc_npy_78 = spctrgrm.numpy()

#Get spectrogram path (where it will be saved)
filename = audio_orig_path.split('/')[-1]
spctrgrm_path = dst_dir_path + '/' + filename[:-4] + '.pt'

#Save spectrogram and save information in new transcript
torch.save(spctrgrm, spctrgrm_path)
