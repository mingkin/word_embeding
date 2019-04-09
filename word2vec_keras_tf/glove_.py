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
    '遍历列'
    for n in range(len(word_set)):
        '遍历行'
        for i in range(len(word_set)-win):
            s1 = word_set[i:i+win]
            '窗口共现统计'
            count = [0 for i in range(win)]
            for j in s1:
                for k in vocab:
                    if k in s1:
                        n = s1.index(k)
                        count[n] = count[n]+1
            k = 0
            print(count)
            print(i)
            for m in range(len(count)):
                coco[n,i+k]=count[m]
                k+=1
    return coco



def build_co_current1(word_set, vocab, win=3):
    coco = np.eye(len(word_set))
    '遍历列'
    for n in range(len(word_set)):
        '遍历行'
        for i in range(0,len(word_set)-win,win):
            s1 = word_set[i:i+win]
            '窗口共现统计'
            count = [0 for i in range(win)]
            for j in s1:
                for k in range(len(vocab)):
                    '''
                    判断元素是否在窗口内，并判断列元素是否在左窗口和右窗口
                    '''
                    if vocab[k] in s1:
                        if word_set[n] in vocab[max(0, k-win):k]:
                            n = s1.index(vocab[k])
                            count[n] = count[n]+1
                        if word_set[n] in vocab[k+1:min(i + 1 + win, len(vocab))]:
                            n = s1.index(vocab[k])
                            count[n] = count[n]+1

            k = 0
            print(count)
            print(i)
            for m in range(len(count)):
                coco[n,i+k]=count[m]
                k+=1
    return coco


def build_co_current2(word_set, vocab, win=3):
    cooc = list()
    counter = 1
    # For each term present in the vocabulary
    for term in word_set:
        vectors = list()
        # Find all the indices of the current term wihtin the ordered list of words
        for sentences in vocab:
            indices = [i for i, x in enumerate(sentences) if x == term]

            vector = [0 for j in range(0, len(word_set))]

            # Find all left and right words upto specified count
            for i in indices:
                leftWords = sentences[max(0, i - win):i]
                rightWords = sentences[i + 1:min(i + 1 + win, len(sentences))]

                increament = len(leftWords)
                for word in leftWords:
                    try:
                        # We skip the word if it is same as the word we are checking for
                        if word_set.index(word) != i:
                            vector[word_set.index(word)] += (1.0 / increament)
                        # Since we are moving from left to right(towards the center word), distance decreases
                        increament -= 1
                    except:
                        increament -= 1

                increament = 1
                for word in rightWords:
                    try:
                        if word_set.index(word) != i:
                            vector[word_set.index(word)] += (1.0 / increament)
                        # Since we are moving from center to right, distance increases
                        increament += 1
                    except:
                        increament += 1
            # Appending each row to the matrix
            vectors.append(vector)

        # We could have many rows for the same term as it may occur many times in the corpus. We need to sum each column and create a resulting vector
        temp = list()
        for i in range(len(word_set)):
            m = 0
            # Adding up each column
            for j in range(len(vectors)):
                m += vectors[j][i]
            # Appending each column to the list
            temp.append(m)
        # Appedning the resulting vector to the co-occurence matrix
        cooc.append(temp)
        print(term + "\t" + str(len(word_set) - counter))
        counter += 1

    # Now extract all the non-zero elements to form a dense matrix
    X = list()
    for i in range(len(word_set)):
        for j in range(len(word_set)):
            if cooc[i][j] != 0:
                X.append([i, j, cooc[i][j]])
    return X,cooc









a = [['a','b','c','d','ef','as','ss','a','e','c','b','f'],
     ['a', 'b', 'c', 'd', 'ef', 'as', 'ss', 'a', 'e', 'c', 'b', 'f']]

a1 = ['a', 'b', 'c', 'd', 'ef', 'as', 'ss', 'a', 'e', 'c', 'b', 'f']
b = []
for i in a:
    for j in i:
        b.append(j)
print(b)
b1 = [i for i in set(b)]
b2 = [i for i in set(a1)]
c = build_co_current1(b2, a1, win=3)

print(np.array(c))

















