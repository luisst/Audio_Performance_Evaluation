#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 04:27:26 2021

@author: luis
"""

import numpy as np

g1 = ['a',
      'aɪ']

g2 = ['b',
      'd',
      'p',
      't',
      'β',
      'ð',
      'pː'] #I added this one, from words like septimo and aceptando.

g3 = ['e',
      'eɪ',
      'ɛ']

g4 = ['f']
g6 = ['k']
g7 = ['l']
g8 = ['tʃ']
g9 = ['u']
g10 = ['w']
g11 = ['x']

g16 = ['m',
      'n',
      'ŋ',
      'ɲ',]

g17 = ['o',
      'oɪ',
      'ɔ']

g18 = ['r',
      'ɾ']

g19 = ['s']

g20 = ['i',
      'ʝ',
      'j']

g21 = ['ɡ', 'ɣ']

#These two are new
g22 = ['eʊ'] #From words like reunir, reunion, etc.
g23 = ['aʊ'] #From words like aunque, audaz, aun, etc.

all_groups_list = [g1, g2, g3, g4,
                   g6, g7, g8, g9, g10,
                   g11,
                   g16, g17, g18, g19, g20,
                   g21, g22, g23]

all_kw_list = ['uno',
                'dos',
                'tɾes',
                'kwatɾo',
                'sinko',
                'sɛɾɔ',
                'kɔmputaðɔɾa',
                'numɛɾɔ']

offset_coef = 1
nummatch_coef = 1
charmatch_coef = 2
difflength_coef = 1

th_others = 2
exact_val = 1
group_val = 0.7


def kw_mapping(input_mapping_phrase, verbose_flag):

    input_pred_phrase_list = " ".join(input_mapping_phrase.split()).split(' ')
    kw_dict = {}
    
    for idx_single_pred in range(0, len(input_pred_phrase_list)):
        
        single_word = input_pred_phrase_list[idx_single_pred]
        
        if len(single_word) > 1:
            kw_match_val_list = [0] * len(all_kw_list)
            # if verbose_flag:
                # print("----------------------------------------------------------------")
        
            # Iterate over all keywords
            for idx_kw in range(0,len(all_kw_list)):
                kw_api_single = all_kw_list[idx_kw]
                
                # print("--------------------------------------------------")
                # print("Current pred: {}  -  KW: {}".format(single_word, kw_api_single))
                # print("--------------------------------------------------")
               
                kw_as_list_ext = ['$'] * (len(single_word)-1) + \
                                list(kw_api_single) + \
                                ['$'] * (len(single_word)-1)
                    
                current_word_match_value = 0
                max_char_found = 0
                max_charmatch_val = 0
                
                # shift char by char single_word
                for idx_c in range(0, (len(single_word) + len(kw_api_single) -1)): 
                    
                    n_char_found = 0
                    match_val_per_slice = 0
                    kw_slice = kw_as_list_ext[idx_c:(idx_c + len(single_word))]
                    # print("----------------------------------------")
                    # print(" KWslice {}".format(kw_slice))
                    
                    # iterate char by char in selectd slices
                    for idx_slice in range(0, len(single_word)):
                        single_char = single_word[idx_slice]
                        search_char = kw_slice[idx_slice]
                        
                        # print("Compare {} - {}".format(single_char, search_char))
                        
                        # iterate in all groups of phonemes
                        for idx_search_char in range(0, len(all_groups_list)):
                            # check character in KW_slice is empty
                            if search_char != '$':
                                # search in all groups where kw char belongs
                                if search_char in all_groups_list[idx_search_char]:
                                    # verify if the single_char is in current kw group
                                    if single_char in all_groups_list[idx_search_char]:
                                        # print('     >>> Found character {} in {}'.format(single_char, all_groups_list[idx_search_char]))
                                        n_char_found = n_char_found + 1
                                        # check if it's the exact character 
                                        if single_char == search_char:
                                            match_val_per_slice = match_val_per_slice + exact_val
                                        else:
                                            match_val_per_slice = match_val_per_slice + group_val
                
                    
                    if n_char_found > max_char_found:
                        max_char_found = n_char_found
                    if match_val_per_slice > max_charmatch_val:
                        max_charmatch_val = match_val_per_slice
                
                # print("(max slice) Value of match: ", max_charmatch_val)
                nummatch_val = max_char_found/len(single_word)
                difflength_val = 1 - abs(len(single_word) - len(kw_api_single))/max(len(single_word), len(kw_api_single))
                
                
                if nummatch_val > 0:
                    dlen_coef = difflength_coef
                else:
                    dlen_coef = 0 
                    
                if difflength_val > 0.3:
                    del_coef = 1
                else:
                    del_coef = 0
                
                current_word_match_value = np.round(del_coef*(charmatch_coef*max_charmatch_val/len(single_word) + dlen_coef*difflength_val), 2)
                # current_word_match_value = np.round(charmatch_coef*max_charmatch_val + dlen_coef*difflength_val, 2)
                # print(' - - {} - {} - char {:.2f} - difflen {:.2f} - normC {:.2f}'.format(single_word, kw_api_single, max_charmatch_val, dlen_coef*difflength_val, charmatch_coef*max_charmatch_val/len(single_word)))
                
                kw_match_val_list[idx_kw] = current_word_match_value
                # print(" - - - - ")
                # if verbose_flag:
                #     print("pred: {} - KW {} - value {}".format(single_word, kw_api_single, current_word_match_value))
            
            if verbose_flag:
                print("-----------------------------------------")
                print("pred: {}  -  KW values {}".format(single_word, kw_match_val_list))
            
            max_val_kw = max(kw_match_val_list)
            max_index_kw = kw_match_val_list.index(max_val_kw)
            
            # print('Prediction val: ', max_val_kw)
            if max_val_kw < th_others:
                kw_found = 'Others'
            else:
                kw_found = all_kw_list[max_index_kw]
            
            if verbose_flag:
                print("{}  -  KW {}".format(single_word, kw_found))
            kw_dict[single_word] = [kw_found, max_val_kw]
            # kw_values_dict[single_word] = max_val_kw
        
    return kw_dict

def kw_classifier(input_phrase, verbose_flag=False):
    
    list_kw_final = []
    list_kw_val_final = []
    if input_phrase != '':
        map_dict = kw_mapping(input_phrase, verbose_flag)
        list_of_values_full = list(map_dict.values())
        list_k = []
        list_v = []
        for idx_k in range (0, len(list_of_values_full)):
            list_k.append(list_of_values_full[idx_k][0])
            list_v.append(list_of_values_full[idx_k][1])
            
        #Print list of keywords values
        for idx_ele in range(0, len(list_k)):
            if list_k[idx_ele] != 'Others':
                list_kw_final.append(list_k[idx_ele])
                list_kw_val_final.append(list_v[idx_ele])
                
        # If all are Others, print list with others
        # if len(list_of_values) == 0:
            # print('List empty, Others!')
            
    return list_kw_final, list_kw_val_final
    