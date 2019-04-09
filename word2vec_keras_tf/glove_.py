# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/4/8 0008 下午 5:17
"""

from word_segment import read_file_seg,word_fre_glov,word_index_word
import numpy as np



'step1 : read corpus'
path = './corpus/test.txt'
vocab, sentence = read_file_seg(path, flag=True)

print(len(vocab))
print(sentence)

def word_signal(vocab):
    word_set = []
    for i in vocab:
        if i in word_set:
            pass
        else:
            word_set.append(i)
    return word_set

word_set = word_signal(vocab)
print(word_set)
print(len(word_set))

word_fre, vocab_size = word_fre_glov(vocab)
word2id, id2word = word_index_word(word_fre)
print(word2id, id2word)
print(len(word2id.keys()))

'step2 : co-currence matrix'

def encode_word(word_set,vocab,word2id):
    m = 0
    word_s = []
    vocav_e = []
    for i in word_set:
        if i in word2id.keys():
            word_s.append(word2id[i])
    for j in vocab:
        if j in word2id.keys():
            word_s.append(word2id[j])
    return word_s, vocav_e

word_s, vocav_e = encode_word(word_set,vocab,word2id)


def build_co_current(word_set, vocab, win=3):
    coco = np.eye(len(word_set))

    for i in range(len(word_set)):
        s1 = word_set[i:i+win]
        for j in s1:
            for k in vocab:
                if k in s1:
                    n = s1.index(k)

























