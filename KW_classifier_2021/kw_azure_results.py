import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score
import unidecode
from num2words import num2words

from phonemizer import phonemize
from phonemizer.separator import Separator


# all_kw_list = ['uno',
#                 'dos',
#                 'tɾes',
#                 'kwatɾo',
#                 'sinko',
#                 'sɛɾɔ',
#                 'kɔmputaðɔɾa',
#                 'numɛɾɔ']

all_kw_ascii_list = ['uno',
                'dos',
                'tres',
                'cuatro',
                'cinco',
                'cero',
                'computadora',
                'numero']


def check_binary(string) :

    # set function convert string
    # into set of characters .
    p = set(string)

    # declare set of '0', '1' .
    s = {'0', '1'}

    if s == p or p == {'0'} or p == {'1'}:
        return True
    else :
        return False


def std_my_phrases(my_input_str):

    if my_input_str == 'nan':
        return " "
    else:
        # remove question marks and commas
        line_text = my_input_str

        line_text = line_text.replace('-', 'minus')
        line_text = line_text.replace('=', 'equal')
        line_text = line_text.replace('+', 'and')
        line_text = line_text.replace('¿', '')
        line_text = line_text.replace('?', '')
        line_text = line_text.replace('-', '')
        line_text = line_text.replace("'", '')
        line_text = line_text.replace(':', '')
        line_text = line_text.replace('/', '')
        line_text = line_text.replace('//', '')
        line_text = line_text.replace('.', ' ')
        line_text = line_text.replace(',', ' ')
        line_text = line_text.replace('[', '')
        line_text = line_text.replace(']', '')
        line_text = unidecode.unidecode(line_text)
    
        # to lower case
        std_text_input = line_text.lower()
        
        # analize each word if binary or not    
        std_list_output = []
        for current_word in std_text_input.split(' '):
            binary_words = []
            if check_binary(current_word):            
                for current_digit in list(current_word):
                    binary_words.append(num2words(current_digit))
                processed_word = ' '.join(binary_words)
            elif current_word.isdigit():
                processed_word = num2words(current_word)
            else:            
                processed_word = current_word
            std_list_output.append(processed_word)
        
        joined_output_words = ' '.join(std_list_output)
        # print(joined_output_words)
        
        # remove extra blanc spaces
        final_word = " ".join(joined_output_words.split())
        return final_word


# def cer_phonemes(my_gt, my_pred, lang):
#     c_sep = Separator(phone='_', syllable='', word=' ') #custom separator
    
#     #Parameters for phonemizer
#     if lang == 'english':
#         L = 'en-us'
#     elif lang == 'spanish':
#         L = 'es-la'
#     else:
#         print(" error ")
#         #language 'es-la' latin-america, 'es' spain
#     B_E = 'espeak' #back end

#     my_gt_phones = phonemize(my_gt, L, B_E, c_sep)[:-2]
#     my_pred_phones = phonemize(my_pred, L, B_E, c_sep)[:-2]
#     print(my_gt_phones)
#     print(my_pred_phones)
#     phones_prob = cer(my_gt_phones, my_pred_phones)
#     return phones_prob


if __name__ == "__main__":       
    # read CSV
    csv_input_path = r'/home/luis/Dropbox/CVpaper/azure_res_516.csv'
    df_all = pd.read_csv(csv_input_path)


    gt_kw_list = []
    y_labels = []

    pred_kw_list = []
    y_pred = []

    cnt = 0

    for index, row in df_all.iterrows():
        my_gt = std_my_phrases(str(row['groundtruth']))
        my_pred_spanish = std_my_phrases(str(row['azurePrediction']))

        # Search for keywords in the GT phrase
        c_gt_list = my_gt.split(" ")
        gt_kw_row = []
        for idx_gt_words in range(0, len(c_gt_list)):
            c_word_gt = c_gt_list[idx_gt_words]
            if c_word_gt in all_kw_ascii_list:
                gt_kw_row.append(c_word_gt)
        gt_kw_list.append(gt_kw_row)

        if len(gt_kw_row) == 0:
            lbl_gt = int(8)
            y_labels.append(lbl_gt)
        else:
            c_gt_lbl = gt_kw_row[0]
            lbl_gt = int(all_kw_ascii_list.index(c_gt_lbl))
            y_labels.append(lbl_gt)


        if my_pred_spanish == 'nan':
            google_pred = ""
        else:
            google_pred = my_pred_spanish

        # Search for keywords in the GT phrase
        google_pred_list = google_pred.split(" ")
        google_kw_row = []
        for idx_google_words in range(0, len(google_pred_list)):
            google_word_gt = google_pred_list[idx_google_words]
            if google_word_gt in all_kw_ascii_list:
                google_kw_row.append(google_word_gt)
        pred_kw_list.append(google_kw_row)

        # Labels Pred
        if len(google_kw_row) == 0:
            lbl_pred = int(8)
            y_pred.append(lbl_pred)
        else:
            google_single_pred = google_kw_row[0]
            lbl_pred = int(all_kw_ascii_list.index(google_single_pred))
            y_pred.append(lbl_pred)

        # print('{}  GT: {}({})  -  {}({})'.format(cnt, 
                                                    # gt_kw_row, 
                                                    # lbl_gt, 
                                                    # google_kw_row, 
                                                    # lbl_pred))
        cnt = cnt + 1   
       
        
    # Accuracy Score (Sum of diagonal/total)
    acc_score = np.round(accuracy_score(y_labels, y_pred), 2)
    print("\nGeneral Accuracy ", acc_score)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_labels, y_pred, 
                                    labels = [0,
                                              1,
                                              2,
                                              3,
                                              4,
                                              5,
                                              6,
                                              7,
                                              8])
    print('\n', conf_matrix)
    
    # Balanced Accuracy Score (average of recall obtained on each class)
    bal_acc_score = np.round(balanced_accuracy_score(y_labels, y_pred), 2)
    print("\nBalanced Accuracy ", bal_acc_score)
        
    # # Print number of Others in the Test Set
    # print(f" Others GT --> {y_labels.count(8)}")
    # print(f" Others Pred --> {y_pred.count(8)}")
    # print('\n')
    
    # for idx_acc in range(0,len(all_kw_ascii_list) + 1):
    #     gt_temp = np.sum(conf_matrix[idx_acc,:])
    #     if gt_temp == 0:
    #         print("Acc {}: {:.2f}".format(idx_acc, 0))
    #     else:    
    #         print("Acc {}: {:.2f}".format(idx_acc, conf_matrix[idx_acc, idx_acc]/gt_temp))
    


########################################

    print("\nRecall global {:.2f} \n".format(recall_score(y_labels, y_pred, average='macro')))
    # print("Recall All{} \n".format(recall_score(y_labels, y_pred, average=None)))
    
    print("Precision global {:.2f} \n".format(precision_score(y_labels, y_pred, average='macro')))
    # print("Precision All{} \n".format(precision_score(y_labels, y_pred, average=None)))
    
    print("F1 global {:.2f} \n".format(f1_score(y_labels, y_pred, average='macro')))
    # print("F1 All{} \n".format(f1_score(y_labels, y_pred, average=None)))
