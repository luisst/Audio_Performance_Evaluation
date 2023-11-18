#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 14:08:23 2021

@author: luis
"""

# from phonemizer import phonemize
# from phonemizer.separator import Separator

import numpy as np

# g1 = ['',
#       '',
#       '']

g1 = ['a',
      'aɪ']

g2 = ['b',
      'd',
      'p'
      't',
      'β',]

g3 = ['e',
      'eɪ']

g4 = ['f']
g5 = ['j']
g6 = ['k']
g7 = ['l']
g8 = ['tʃ']
g9 = ['u']
g10 = ['w']
g11 = ['x']
g12 = ['ð']
g13 = ['ɛ']
g14 = ['ɡ']
g15 = ['ɣ']

g16 = ['m',
      'n',
      'ŋ',
      'ɲ',]

g17 = ['o',
      'oɪ']

g18 = ['r',
      'ɾ']

g19 = ['s',
      'ɔ']

g20 = ['i',
      'ʝ']

all_groups_list = [g1, g2, g3, g4, g5,
                   g6, g7, g8, g9, g10,
                   g11, g12, g13, g14, g15,
                   g16, g17, g18, g19, g20]


# all_kw_list = ['numɛɾɔ',
#                'wxmɾaɔ',
#                'sr',
#                'ziəɹoʊ',
#                'numɔɾɛ']

all_kw_list = ['uno',
                'dos',
                'tɾes',
                'kwatɾo',
                'sinko',
                'sɛɾɔ',
                'kɔmputaðɔɾa',
                'numɛɾɔ',
                'numɛɾɔs',
                'binaɾjo']

offset_coef = 1
nummatch_coef = 1
charmatch_coef = 1
difflength_coef = 1

th_others = 2.2
exact_val = 1
group_val = 0.7

verbose_flag = False

# single_word = 'ziəɹoʊ'
# single_word = "merɔ"
# single_word = "sɛɾɔ"
# single_word = "sɛrɔ"

# compare word by word in input phrase
input_pred_phrase = 'ziəɹoʊ merɔ sɛɾɔ'

kw_dict = {}
input_pred_phrase_list = input_pred_phrase.split(" ")

for idx_single_pred in range(0, len(input_pred_phrase_list)):
    
    single_word = input_pred_phrase_list[idx_single_pred]
    kw_match_val_list = [0] * len(all_kw_list)
    if verbose_flag:
        print("----------------------------------------------------------------")

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
        difflength_val = abs(len(single_word) - len(kw_api_single))/max(len(single_word), len(kw_api_single))
        
        
        if nummatch_val > 0:
            dlen_coef = difflength_coef
        else:
            dlen_coef = 0 
        
        current_word_match_value = np.round(charmatch_coef*max_charmatch_val + dlen_coef*difflength_val, 2)
        kw_match_val_list[idx_kw] = current_word_match_value
        # print(" - - - - ")
        if verbose_flag:
            print("pred: {} - KW {} - value {}".format(single_word, kw_api_single, current_word_match_value))
    
    if verbose_flag:
        print("-----------------------------------------")
        print("pred: {}  -  KW values {}".format(single_word, kw_match_val_list))
    
    max_val_kw = max(kw_match_val_list)
    max_index_kw = kw_match_val_list.index(max_val_kw)
    
    if max_val_kw < th_others:
        kw_found = 'Others'
    else:
        kw_found = all_kw_list[max_index_kw]
    
    print("pred: {}  -  KW {}".format(single_word, kw_found))
    kw_dict[single_word] = kw_found