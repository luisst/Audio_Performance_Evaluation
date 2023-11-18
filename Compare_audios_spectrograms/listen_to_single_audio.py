#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:16:59 2022

@author: luis
"""
import numpy as np
import sounddevice as sd

fs=16000


def listen_1_audio(full_npy, idx_audio):
    audio_1 = full_npy[idx_audio]
    audio_1 = np.trim_zeros(audio_1)
    sd.play(audio_1, fs)


npy_path = '/home/luis/Documents/TS_EN_phrases/EN_filtered_ALL/EN_full70K_gt_ext_x4.npy'

my_dataset_GT = np.load(npy_path, allow_pickle=True)

listen_1_audio(my_dataset_GT, 0)