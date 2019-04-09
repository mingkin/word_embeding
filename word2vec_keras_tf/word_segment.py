# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/4/8 0008 上午 10:30
"""

import jieba
import collections
from tqdm import tqdm
import numpy as np
import math



def read_file_seg(path, flag=True):
    word = []
    stence = []
    if flag == True:
        with open(path, 'r', encoding='utf8') as f:
            num = 0
            for i in f.readlines():
                word.extend([j for j in jieba.cut(i.strip('\n'), cut_all=False)])
                stence.append([j for j in jieba.cut(i.strip('\n'), cut_all=False)])
                print('current line :%d' % num)
                num = num + 1
        return word, stence
    else:
        with open(path, 'r', encoding='utf8') as f:
            num = 0
            for i in f.readlines():
                word.extend(i.strip('\n'))
                stence.append(i)
                print('current line :%d' % num)
                num = num + 1
        return word, stence


def read_file_seg_big(path, flag=True):
    if flag == True:
        with open(path, 'r', encoding='utf8') as f:
            for i in f.read():
                yield [j for j in jieba.cut(i.strip('\n'), cut_all=False)]

    else:
        with open(path, 'r', encoding='utf8') as f:
            for i in f.read():
                yield i.strip('\n')



def word_frequence(words, fre =10, path = './corpus/word_fre.txt'):
    #vocab_size = 50000
    #vocab = collections.Counter(words).most_common(vocab_size - 1)
    vocab = collections.Counter(words)
    word_fre = [['UNK', 0]]
    with open(path, 'w', encoding='utf8') as f:
        for k, v in vocab.items():
            if v >= fre:
                word_fre.append([k, v])
                f.write(str(k))
                f.write('\t')
                f.write(str(v))
                f.write('\n')
            else:
                pass
    vocab_size = len(word_fre)
    return word_fre, vocab_size


def word_fre_glov(words, path='./corpus/word_fre_glove.txt'):
    #vocab_size = 50000
    #vocab = collections.Counter(words).most_common(vocab_size - 1)
    vocab = collections.Counter(words)
    word_fre = []
    with open(path, 'w', encoding='utf8') as f:
        for k, v in vocab.items():
            word_fre.append([k, v])
            f.write(str(k))
            f.write('\t')
            f.write(str(v))
            f.write('\n')
    vocab_size = len(word_fre)
    return word_fre, vocab_size


def word_index_word(word_fre):
    word_index = {}
    index_word = {}
    j = 0
    for i in word_fre:
        word_index[i[0]] = j
        index_word[str(j)] = i[0]
        j = j+1
    return word_index, index_word


def corups_index(vocab, word_index):
    d = []
    for i in vocab:
        s = []
        for j in i:
            if j in word_index.keys():
                s.append(word_index[j])
            else:
                s.append(0)
        d.append(s)
    return d



def split_data(data, win=3):
    X_train = []
    Y_train = []
    for i in tqdm(range(len(data))):
        d = data[i]
        for j in range(len(d)):
            if j+win >len(d):
                pass
            else:
                s1 = d[j:j+win]
                #print(s1)
                X_train.extend([s1[math.floor(len(s1)/2)] for i in range(win-1)])
                #print([s1[math.floor(len(s1)/2)] for i in range(win-1)])
                Y_train.extend(s1[:math.floor(len(s1)/2)])
                #print(s1[:math.floor(len(s1)/2)])
                Y_train.extend(s1[math.floor(len(s1)/2)+1:])
                #print(s1[math.floor(len(s1)/2)+1:])
        #print(len(X_train), len(Y_train))
    X_train = np.array(X_train)
    Y_train = np.array(Y_train).reshape((X_train.shape[0],1))
    print(X_train.shape, Y_train.shape)
    return X_train, Y_train



