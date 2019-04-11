# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/4/8 0008 下午 5:17
"""

from word_segment import read_file_seg, word_fre_glov, word_index_word,encode_word
import numpy as np
import random, math
from math import log


'step1 : read corpus'
path = './corpus/test.txt'
vocab, sentence = read_file_seg(path, flag=True)


word = [i for i in set(vocab)]
word_fre, vocab_size = word_fre_glov(vocab)
word2id, id2word = word_index_word(word_fre)
print(word2id, id2word)
word_s, vocav_e = encode_word(word,vocab,word2id)

'step2 : co-currence matrix'



def build_co_current(word, vocab, win=3):
    '''
    :param word: 所有不重复词的list
    :param vocab: 所有句子分词后的句子按顺序写到一个list[s1,s2,...]
    :param win: 窗口
    :return: 共现矩阵
    '''

    coco = np.eye(len(word))
    '遍历列'
    for i in range(len(word)):
        for j in range(len(word)):
            if i ==j:
                '自己与自己共现跳过'
                continue
            else:
                '窗口共现统计'
                count = 0
                count_l = 0
                count_r = 0
                '''判断元素是否在窗口内，并判断列列元素是否在左窗口和右窗口'''
                for k, v in enumerate(vocab):
                    if v == word[i]:
                        '距离衰减 1/d'
                        #print(word[j])
                        if word[j] in vocab[max(0, k-win):k]:
                            #print('zuo', vocab[max(0, k-win):k])
                            c_l = [k for k, v in enumerate(vocab[max(0, k-win):k]) if v == word[j]]
                            if len(c_l) > 1:
                                s_l = []
                                for m in c_l:
                                    d_l = 1/int(win-m)
                                    count_l += 1
                                    s_l.append(count_l * d_l)
                                count = sum(s_l)
                            else:
                                d_l = 1 / int(win-int(c_l[0]))
                                count_l += 1
                                count = count_l*d_l
                        elif word[j] in vocab[k+1: min(k + 1 + win, len(vocab))]:
                            #print('you',vocab[k+1: min(k + 1 + win, len(vocab))])
                            c_r = [k for k, v in enumerate(vocab[k+1: min(k + 1 + win, len(vocab))]) if v == word[j]]

                            if len(c_r) > 1:
                                s_r = []
                                for n in c_r:
                                    d_r = 1 / int(n+1)
                                    count_l += 1
                                    s_r.append(count_r * d_r)
                                count = sum(s_r)
                            else:
                                d_r = 1 / int(int(c_r[0])+1)
                                count_l += 1
                                count = count_l * d_r
                        else:
                            pass
            coco[i, j] = count
    return coco



'step3: train_glove'

def glove_fit(word, co_matrix, iterations, vec_dim, lr, xMax=100, alpha=0.75):
    '''
    :param : word
    :param co_matrix
    :param iterations:训练次数
    :param vec_dim
    :param lr:
    :param xMax:  weight_func xmax取值都是100
    :param alpha: α 的取值都是0.75
    :return:
    '''

    globalCost = 0
    #初始化
    W = [[random.uniform(-0.5, 0.5) for i in range(vec_dim)] for j in range(len(word))]
    biases = [random.uniform(-0.5, 0.5) for i in range(len(word))]
    # Training is done via adaptive gradient descent (AdaGrad).
    # To make this work we need to store the sum of squares of all previous gradients.
    # Initialize the squared gradient weights and biases to 1
    gradSquaredW = [[1 for i in range(vec_dim)] for j in range(len(word))]
    gradSquaredBiases = [1 for i in range(len(word))]

    for n in range(iterations):
        for i in range(len(word)):
            for j in range(len(word)):
                if i != j:
                    co_fre = co_matrix[i][j]
                    w1 = W[i]
                    w2 = W[j]
                    '防止log出现错误'
                    if co_fre !=0.0:
                        # Weighting function
                        if co_fre < xMax:
                            f = (co_fre / xMax) ** alpha
                        else:
                            f = 1
                        innerCost = (np.dot(np.array(w1), np.array(w2)) + biases[i] + biases[j] + log(co_fre))

                        # Calculate cost
                        cost = f * (innerCost ** 2)

                        globalCost += 0.5 * cost

                        # Calculate the gradient for the word as both main and contextual
                        gradMain = f * np.dot(innerCost, np.array(w2))
                        gradContext = f * np.dot(innerCost, np.array(w1))

                        # Calculate the gradient of the bias for the word
                        gradBiasMain = f * innerCost
                        gradBiasContext = f * innerCost

                        # Applying adagrad
                        for a in range(vec_dim):
                            w1[a] -= ((gradMain[a] * lr) / math.sqrt(sum(gradSquaredW[i])))
                            gradSquaredW[i][a] += gradMain[a] ** 2

                        for a in range(vec_dim):
                            w2[a] -= ((gradContext[a] * lr) / math.sqrt(sum(gradSquaredW[j])))
                            gradSquaredW[j][a] += gradContext[a] ** 2

                        biases[i] -= ((lr * gradBiasMain) / math.sqrt(gradSquaredBiases[i]))
                        biases[j] -= ((lr * gradBiasContext) / math.sqrt(gradSquaredBiases[j]))

                        gradSquaredBiases[i] += gradBiasMain ** 2
                        gradSquaredBiases[j] += gradBiasContext ** 2

                        W[i] = w1
                        W[j] = w2
                    else:
                        continue
                else:
                    continue
        print("GloVe--Iteration-" + str(n+1) + "--Cost-" + str(globalCost/(n+1)))

    return W



'step: write_file'

def save_glove_vector(word,vector,path='./model/glove_vector.txt'):
    w_num = len(word)
    v_dim = len(vector[0])
    with open(path, 'w', encoding='utf8') as f:
        f.write(str(w_num)+'*'+str(v_dim))
        f.write('\n')
        for i in range(w_num):
            f.write(str(word[i]))
            f.write('\t')
            f.write(' '.join([str(j) for j in vector[i]]))
            f.write('\n')


def test():
    a = [['a','b','c','d','ef','as','ss','a','e','c','b','f'],
         ['a', 'b', 'c', 'd', 'ef', 'as', 'ss', 'a', 'e', 'c', 'b', 'f']]
    b = []
    for i in a:
        for j in i:
            b.append(j)
    b1 = [i for i in set(b)]
    c1 = build_co_current(b1, b, win=3)
    w2 = glove_fit(b1, c1, 10, 100, 0.01, xMax=100, alpha=0.75)
    save_glove_vector(b1, w2, path='./model/glove_vector.txt')
    print(w2)
    # v1 = vocab[:1000]
    # w1 = [i for i in set(v1)]
    # c1 = build_co_current(w1, v1, win=3)
    # w2 = glove_fit(w1, c1, 10, 100, 0.01, xMax=100, alpha=0.75)












