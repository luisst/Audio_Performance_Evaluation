# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:11:12 2021

@author: luis
"""

import textdistance
import unidecode

from phonemizer import phonemize
from phonemizer.separator import Separator

kw_list_ascii_es = ['uno',
              'dos',
              'tres',
              'cuatro',
              'cinco',
              'cero',
              'computadora',
              'numero',
              'numeros',
              'binario']



B_E = 'espeak' #back end
L = 'es-la'
c_sep = Separator(phone='', syllable='', word=' ') #custom separator

# kw_api_es_single = phonemize(kw_list_ascii_es[5], L, B_E, c_sep)[0:-1]
# print(kw_api_es_single)


# my_pred_single = "z_iə_ɹ_oʊ"

# my_pred_single = "sɛɾɔ"
# kw_api_es_single2 = 'numɛɾɔ'


my_pred_single = "cero"
kw_api_es_single2 = 'numero'

# numɛɾɔ

my_pred_ipa = my_pred_single.replace('_', '')


###########################################################################

hamming_distance_val = textdistance.hamming.normalized_distance(my_pred_ipa, kw_api_es_single2)
print(hamming_distance_val)

hamming_similarity_val = textdistance.hamming.similarity(my_pred_ipa, kw_api_es_single2)
print(hamming_similarity_val)


# try phonemes alphabet

# google1 = "propuesta"

# search for the keyboard between all the words, shift.

# group phonemes that sound similar:
    # get vocabulary from phonemizer
    # get vocabulary from our system

# check the tildes.

# take into account the keyboard size word vs the predicted word. closer the better.

# take into account not only the common characters, but the order, if it's before or after, 
# the similarity can do this? the relantionship of one character after the other.


### to do:
    # try editex https://anhaidgroup.github.io/py_stringmatching/v0.3.x/Editex.html
    # with phonemes and see if i need to convert them to words for it to work.
    # try with group, how to define a group