#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 04:19:00 2021

@author: luis
"""

import pandas as pd
import numpy as np
from kwclassifier import kw_classifier, all_kw_list
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

df_all = pd.read_csv (r'./inferences_AO_SP_516_YesRL2.txt', index_col=0, delimiter='\t')
# !!!! Columns names are wrong, please fix later


gt_kw_list = []
y_labels = []
y_pred = []

per_old_list = []
cnt = 0
# Read results CSV file line by line
for index, row in df_all.iterrows():
    current_per = row['c1']
    current_gt = str(row['WER'])
    current_prediction = str(row['Predicted IPAs'])

    # Append list of old Phoneme Error Rate (PER)
    per_old_list.append(current_per)

    # Eliminate underscore in the API strings
    c_gt = current_gt.replace('_', '')

    # Search for keywords in the GT phrase
    c_gt_list = c_gt.split(" ")
    gt_kw_row = []
    for idx_gt_words in range(0, len(c_gt_list)):
        c_word_gt = c_gt_list[idx_gt_words]
        if c_word_gt in all_kw_list:
            gt_kw_row.append(c_word_gt)
    gt_kw_list.append(gt_kw_row)

    if current_prediction == 'nan':
        c_pred = ""
    else:
        # Eliminate underscore in the API strings
        c_pred = current_prediction.replace('_', '')

    # Run Keyword Classifier Module
    kw_detected_list, kw_detected_vals = kw_classifier(c_pred, verbose_flag = False)

    #### Calculate accuracy (using only one KW)
    # Labels GT
    if len(gt_kw_row) == 0:
        lbl_gt = int(8)
        y_labels.append(lbl_gt)
    else:
        c_gt_lbl = gt_kw_row[0]
        lbl_gt = int(all_kw_list.index(c_gt_lbl))
        y_labels.append(lbl_gt)
    # Labels Pred
    if len(kw_detected_list) == 0:
        lbl_pred = int(8)
        y_pred.append(lbl_pred)
    else:
        c_network_pred = kw_detected_list[0]
        lbl_pred = int(all_kw_list.index(c_network_pred))
        y_pred.append(lbl_pred)

    # print('{}  GT: {}({})  -  {}({}) {}'.format(cnt, gt_kw_row, lbl_gt, kw_detected_list, lbl_pred, kw_detected_vals))
    cnt = cnt + 1



# Accuracy Score (Sum of diagonal/total)
acc_score = np.round(accuracy_score(y_labels, y_pred), 2)
# print("\nGeneral Accuracy ", acc_score)

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_labels, y_pred,
#                                labels = [0,
#                                          1,
#                                          2,
#                                          3,
#                                          4,
#                                          5,
#                                          6,
#                                          7,
#                                          8])
# print('\n', conf_matrix)

# Balanced Accuracy Score (average of recall obtained on each class)
bal_acc_score = np.round(balanced_accuracy_score(y_labels, y_pred), 2)
# print("\nBalanced Accuracy ", bal_acc_score)
#
# # Show Old average PER
# print('\nAverage Old (1 - PER)', ( 1 - np.mean(per_old_list)))

# # Print number of Others in the Test Set
# y_lbl_array = np.array(y_labels)
# y_pred_array = np.array(y_pred)
# print(f" Others GT --> {y_lbl_array.size - np.count_nonzero(y_lbl_array)}")
# print(f" Others Pred --> {y_pred_array.size - np.count_nonzero(y_pred_array)}")
# print('\n')

# for idx_acc in range(0,len(all_kw_list) + 1):
#     gt_temp = np.sum(conf_matrix[idx_acc,:])
#     if gt_temp == 0:
#         print("Acc {}: {:.2f}".format(idx_acc, 0))
#     else:
#         print("Acc {}: {:.2f}".format(idx_acc, conf_matrix[idx_acc, idx_acc]/gt_temp))


# print("\nRecall global: {:.2f} \n".format(recall_score(y_labels, y_pred, average='macro')))

myrecall_values = recall_score(y_labels, y_pred, average=None)
for idx_mimi in range(0, len(myrecall_values)):
    print("Index {}   recal value {}".format(idx_mimi, np.round(myrecall_values[idx_mimi], 1)))

# print("Recall All{} \n".format())

# print("Precision global: {:.2f} \n".format(precision_score(y_labels, y_pred, average='macro')))
# print("Precision All{} \n".format(precision_score(y_labels, y_pred, average=None)))

# print("F1 global: {:.2f} \n".format(f1_score(y_labels, y_pred, average='macro')))
# print("F1 All{} \n".format(f1_score(y_labels, y_pred, average=None)))



# 2) th increase send all data to others, incresing acc, but reducing the false positives and the diagonal

# 4) try a balanced dataset so it's easier to check the results.
# because i have so many others, the diagonal would be hardly filled.

#9) Later: penalize a lot when the difference is significant, but not add when the distance is the same

# include csv from google speech to text and see how many of those have the keywords.
# delete numeros from the keywords list. and binarios.

# count false negatives

# when we dont use a vocabulary, 72 percent when we do.
# report the parameters from our model, with
# achivieng better accuracy,
# getting state of the art performance

# kw that i getting consistently
# include confusion matrix,



###########################

# if kw_detected_list, error for each kw missed

# count false positives, false negatives.

# Balance the test set, otherwise predicting all as Others would be dummy



# To-do:
    # Calculate new metrics and compare with old metrics
    # Organize properly the Spanish phonemes groups in the function.
    # Add support to write a csv file with the results
