"""
Utility function to calculate the feature of a word
"""

import numpy as np

VOWELS = ['a', 'e', 'i', 'o', 'u']

COMMON_MU = 18.48696912587767 
COMMON_SIGMA = 3.340223978055492

# get duplicate score
def get_duplicate_score(word: str):
    ''' Get duplicate score, 0 dup=0, 1 dup=1, 2 dup=2'''
    assert len(word) == 5
    return (5-len(set(word)))

def stats_letter_freq(words):
    ''' Stats the overall latter freq, return a 26-len array'''
    count = np.zeros(26)
    for i in range(len(words)):
        for j in range(5):
            count[ord(words[i][j])-97] += 1
    return count

def get_freq_score(word: str, freq_count: np.ndarray):
    ''' Get freq score, with sqrt sum'''
    assert len(word) == 5
    score = 0
    for i in range(5):
        score += freq_count[ord(word[i])-97]
    return score

def get_freq_score_weighted(word: str, freq_count: np.ndarray):
    ''' Get freq score, first letter has double score, with sqrt sum'''
    assert len(word) == 5
    score = 0
    for i in range(5):
        if i == 0:
            score += freq_count[ord(word[i])-97] * 2
        else:
            score += freq_count[ord(word[i])-97]

    return score ** 0.5

def get_duplicate_conti_score(word: str):
    ''' Get duplicate continued score. '''
    score = 0
    for j in range(4):
        if word[j] == word[j+1]:
            score += 1
    return score

def get_vowel_number_score(word: str):
    ''' Get number of vowels score. '''
    count = 0
    for j in range(5):
        if word[j] in VOWELS:
            count += 1
    return count

def get_begin_vowel_score(word: str):
    ''' Get if vowel at beginning score. '''
    return 1 if word[0] in VOWELS else 0


def get_freq_var_score(word: str, freq_count: np.ndarray):
    ''' Get large if freq's variance is large'''
    arr = np.zeros(5)
    for i in range(5):
        arr[i] = freq_count[ord(word[i])-97]    
    return arr.var()

def sort_dict_value(dictionary: dict):
    ''' Sort dictionary by value '''
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    sorted_value_index = np.argsort(values)
    sorted_value_index = sorted_value_index[::-1]
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
    return sorted_dict


def get_double_count(word_list):
    ''' Count repeated continuous doubles '''
    # frequency er
    double_count = dict()
    for word in word_list:
        for i in range(4):
            if len(word) == 5:
                double = word[i:i+2]
                double_count[double] = double_count.get(double, 0) + 1

    double_count = sort_dict_value(double_count)

    return double_count

def get_trible_count(word_list):
    ''' Count repeated continuous tribles '''
    # frequency ine
    trible_count = dict()
    for word in word_list:
        for i in range(3):
            if len(word) == 5:
                double = word[i:i+3]
                trible_count[double] = trible_count.get(double, 0) + 1

    trible_count = sort_dict_value(trible_count)
    return trible_count

def get_spaced_double_count(word_list):
    ''' Count repeated spaced doubles '''
    # frequency a_e
    doublespace_count = dict()
    for word in word_list:
        for i in range(3):
            if len(word) == 5:
                double = word[i] + "_" + word[i+2]
                doublespace_count[double] = doublespace_count.get(double, 0) + 1

    doublespace_count = sort_dict_value(doublespace_count)
    return doublespace_count

def get_begin_count(word_list):
    ''' Count first letter frequency'''
    # frequency begin alpha
    begin_count = dict()
    for word in word_list:
        # for i in range(3):
        if len(word) == 5:
            begin = word[0]
            begin_count[begin] = begin_count.get(begin, 0) + 1

    begin_count = sort_dict_value(begin_count)

    return begin_count

def get_end_count(word_list):
    ''' Count end letter frequency'''
    # frequency end alpha
    end_count = dict()
    for word in word_list:
        # for i in range(3):
        if len(word) == 5:
            end = word[4]
            end_count[end] = end_count.get(end, 0) + 1

    end_count = sort_dict_value(end_count)

    return end_count



def get_double_score(word: str, double_counts: dict, threshold=5):
    ''' Returns the double frequencys of a word in double_counts '''
    score = 0
    for i in range(4):
        double = word[i:i+2]
        if double_counts.get(double, 0) >= threshold:
            score += double_counts.get(double, 0)
    return score

def get_trible_score(word: str, trible_counts: dict, threshold=2):
    ''' Returns the trible frequencys of a word in double_counts '''
    score = 0
    for i in range(3):
        trible = word[i:i+3]
        if trible_counts.get(trible, 0) >= threshold:
            score += trible_counts.get(trible, 0)
    return score

def get_spaced_double_score(word: str, spaced_double_counts: dict, threshold=2):
    ''' Returns the spaced double frequencys of a word in trible_counts '''
    score = 0
    for i in range(3):
        spaced_double = word[i] + "_" + word[i+2]
        if spaced_double_counts.get(spaced_double, 0) >= threshold:
            score += spaced_double_counts.get(spaced_double, 0)
    return score


def get_begin_score(word: str, begin_counts: dict, threshold=4):
    ''' Returns the end of a word in end_counts '''
    if begin_counts.get(word[0], 0) >= threshold:
        return begin_counts.get(word[0], 0)
    else:
        return 0


def get_end_score(word: str, end_counts: dict, threshold=4):
    ''' Returns the spaced double frequencys of a word in double_counts '''
    if end_counts.get(word[4], 0) >= threshold:
        return end_counts.get(word[4], 0)
    else:
        return 0



