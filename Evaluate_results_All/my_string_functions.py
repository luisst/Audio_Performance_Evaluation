import pandas as pd
from num2words import num2words
import numpy as np
import os
import unidecode

from phonemizer import phonemize
from phonemizer.separator import Separator


def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


def check_binary(string) :

    # set function convert string
    # into set of characters .
    p = set(string)

    # declare set of '0', '1' .
    s = {'0', '1'}

    # check set p is same as set s
    # or set p contains only '0'
    # or set p contains only '1'
    # or not, if any one condition
    # is true then string is accepted
    # otherwise not .
    if s == p or p == {'0'} or p == {'1'}:
        return True
    else :
        return False


def std_my_phrases(my_input_str, empty_counter=0):

    if my_input_str == 'nan':
        empty_counter = empty_counter + 1
        return " ", empty_counter
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
        print(joined_output_words)

        # remove extra blanc spaces
        final_word = " ".join(joined_output_words.split())
        return final_word, empty_counter


def cer_phonemes(my_gt, my_pred, lang):
    c_sep = Separator(phone='_', syllable='', word=' ') #custom separator
    B_E = 'espeak' #back end

    #Parameters for phonemizer
    if lang == 'english':
        L = 'en-us'
        my_gt_phones = phonemize(my_gt, L, B_E, c_sep)[:-2]
        my_pred_phones = phonemize(my_pred, L, B_E, c_sep)[:-2]
        print(my_gt_phones)
        print(my_pred_phones)
        phones_prob = cer(my_gt_phones, my_pred_phones)
        return phones_prob
    elif lang == 'spanish':
        L = 'es-419'
        my_gt_phones = phonemize(my_gt, L, B_E, c_sep)[:-2]
        my_pred_phones = phonemize(my_pred, L, B_E, c_sep)[:-2]
        print(my_gt_phones)
        print(my_pred_phones)
        phones_prob = cer(my_gt_phones, my_pred_phones)
        return phones_prob
    elif lang == 'bilingual':
        my_gt_phones_eng = phonemize(my_gt, 'en-us', B_E, c_sep)[:-2]
        my_pred_phones_eng = phonemize(my_pred, 'en-us', B_E, c_sep)[:-2]
        phones_prob_eng = cer(my_gt_phones_eng, my_pred_phones_eng)

        my_gt_phones_spa = phonemize(my_gt, 'es-419', B_E, c_sep)[:-2]
        my_pred_phones_spa = phonemize(my_pred, 'es-419', B_E, c_sep)[:-2]
        phones_prob_spa = cer(my_gt_phones_spa, my_pred_phones_spa)

        phones_prob_bilingual = min(phones_prob_eng, phones_prob_spa)
        return phones_prob_bilingual
    else:
        print(" error ")
        return 1
        #language 'es-la' latin-america, 'es' spain

