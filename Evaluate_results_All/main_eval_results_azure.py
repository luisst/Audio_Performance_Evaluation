#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 00:18:09 2022

@author: luis
"""

import pandas as pd
import sys
import numpy as np
import pathlib
sys.path.append("./../")
from my_files_utils import *

path_this_file = str(pathlib.Path().resolve())

from my_string_functions import std_my_phrases, cer, cer_phonemes, wer


# csv_input_path = r'/home/luis/Dropbox/ResearchJanuary/march6GT/GT_for_test_Testset/aolme_testset_spanish_results.csv'
# csv_input_path = r'/home/luis/Dropbox/dataset_AOLME_pipelineLuis/Aolme2_spanish_Testset/flac_audios/transcript_google_results.csv'

csv_input_path = r'/home/luis/Dropbox/SpeechSpring2022/libriSpeechTest_transcript_Azure_merged_2022-06-24-20_36.csv'

# lang_mode = 'spanish'
# lang_mode = 'english'
lang_mode = 'bilingual'

input_csv_name = csv_input_path.split('/')[-1].split('.')[0]
log_result_path = '/'.join(csv_input_path.split('/')[:-1]) + '/' + 'logs2_cer_wer_' + input_csv_name + '.txt'

df_all = pd.read_csv(csv_input_path)
cer_list = []
wer_list = []
cer_list_phonemes = []
cnt_empty_preds = 0

msg = "Read csv file {}\n ------------------------------ \n".format(input_csv_name)
log_message(msg, log_result_path, 'w', False)

for index, row in df_all.iterrows():
    my_gt, _ = std_my_phrases(str(row['transcript']), 'GT', log_result_path)
    my_pred, cnt_empty_preds = std_my_phrases(str(row['azure']), 'PD', log_result_path, empty_counter=cnt_empty_preds)
    current_path  = str(row['path'])

    current_name_noExtension = current_path.split('/')[-1].split('.')[0]

    current_lang = current_name_noExtension[-11:-4]

    my_prob = cer(my_gt, my_pred)
    
    wer_prob = wer(my_gt, my_pred)

    # prob_phonemes = cer_phonemes(my_gt, my_pred, current_lang, log_result_path)

    cer_list.append(my_prob)
    wer_list.append(wer_prob)
    # cer_list_phonemes.append(prob_phonemes)

    msg = "cer: {:.2f} | wer: {:.2f}\n -------------------------------------- \n".format(my_prob, wer_prob)
    # msg = "prob: {} \n prob_ph: {} \n -------------------------------------- \n".format(my_prob, prob_phonemes)
    log_message(msg, log_result_path, 'a')

cer_avg = np.average(np.array(cer_list))
wer_avg = np.average(np.array(wer_list))
# cer_avg_phonemes = np.average(np.array(cer_list_phonemes))

msg = "Plain text: {:.2f}\n".format(1 - cer_avg)
log_message(msg, log_result_path, 'a')

msg = "Plain text WER : {:.2f} | Inverse: {:.2f}\n".format(wer_avg, 1 - wer_avg)
log_message(msg, log_result_path, 'a')

# msg = "Phonemes: {:.2f}\n".format(1 - cer_avg_phonemes)
# log_message(msg, log_result_path, 'a')


empty_perct = (cnt_empty_preds/len(cer_list))*100
msg = "Empty predictions: {}   |   percentage: {:.2f}%".format(cnt_empty_preds, empty_perct)
log_message(msg, log_result_path, 'a')
