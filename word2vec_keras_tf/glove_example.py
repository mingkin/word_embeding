# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/4/8 0008 下午 5:32
"""

import numpy as np
import random
from math import log
import math
from scipy import spatial

globalCost = 0


def buildCorpusAndVocab():
    f = open("C:\\Users\\atifh\\Desktop\\masc_wordsense\\data\\Full_set\\round1\\launch-v\\launch-v.txt")
    lines = f.readlines()
    corpus = ""
    for line in lines:
        corpus += line.lower()
    corpus = corpus.replace('(', '')
    corpus = corpus.replace(')', ' ')
    corpus = corpus.replace('"', ' ')
    corpus = corpus.replace("'", "")

    vocab = sorted(list(set(corpus.split())))

    return vocab, corpus


# Creates the co-occurence matrix
def buildCoocM(vocab, corpus, windowSize):
    paras = corpus.split("\n")
    words = list()
    # Words are appended in the order that they are present in the sentences
    for sentences in paras:
        words.append(sentences.split())

    cooc = list()
    counter = 1
    # For each term present in the vocabulary
    for term in vocab:
        vectors = list()
        # Find all the indices of the current term wihtin the ordered list of words
        for sentences in words:
            indices = [i for i, x in enumerate(sentences) if x == term]

            vector = [0 for j in range(0, len(vocab))]

            # Find all left and right words upto specified count
            for i in indices:
                leftWords = sentences[max(0, i - windowSize):i]
                rightWords = sentences[i + 1:min(i + 1 + windowSize, len(sentences))]

                increament = len(leftWords)
                for word in leftWords:
                    try:
                        # We skip the word if it is same as the word we are checking for
                        if vocab.index(word) != i:
                            vector[vocab.index(word)] += (1.0 / increament)
                        # Since we are moving from left to right(towards the center word), distance decreases
                        increament -= 1
                    except:
                        increament -= 1

                increament = 1
                for word in rightWords:
                    try:
                        if vocab.index(word) != i:
                            vector[vocab.index(word)] += (1.0 / increament)
                        # Since we are moving from center to right, distance increases
                        increament += 1
                    except:
                        increament += 1
            # Appending each row to the matrix
            vectors.append(vector)

        # We could have many rows for the same term as it may occur many times in the corpus. We need to sum each column and create a resulting vector
        temp = list()
        for i in range(len(vocab)):
            m = 0
            # Adding up each column
            for j in range(len(vectors)):
                m += vectors[j][i]
            # Appending each column to the list
            temp.append(m)
        # Appedning the resulting vector to the co-occurence matrix
        cooc.append(temp)
        print
        term + "\t" + str(len(vocab) - counter)
        counter += 1

    # Now extract all the non-zero elements to form a dense matrix
    X = list()
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if cooc[i][j] != 0:
                X.append([i, j, cooc[i][j]])
    return X


def gloveTrain(vocab, cooc, iterations, d, learningRate, xMax, alpha):
    global globalCost

    # Initialize some random weights of specified vector size
    W = [[random.uniform(-0.5, 0.5) for i in range(d)] for j in range(2 * len(vocab))]
    # Initialize some random biases
    biases = [random.uniform(-0.5, 0.5) for i in range(2 * len(vocab))]

    # Training is done via adaptive gradient descent (AdaGrad). To make this work we need to store the sum of squares of all previous gradients.
    # Initialize the squared gradient weights and biases to 1
    gradSquaredW = [[1 for i in range(d)] for j in range(2 * len(vocab))]
    gradSquaredBiases = [1 for i in range(2 * len(vocab))]

    for i in range(iterations):
        # random.shuffle(cooc)
        for mainID, contextID, data in cooc:
            w1 = W[mainID]
            w2 = W[contextID + len(vocab)]
            x = data

            # Weighting function
            if x < xMax:
                f = (x / xMax) ** alpha
            else:
                f = 1

            innerCost = (np.dot(np.array(w1), np.array(w2)) + biases[mainID] + biases[contextID + len(vocab)] + log(
                data))
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
            for a in range(d):
                w1[a] -= ((gradMain[a] * learningRate) / math.sqrt(sum(gradSquaredW[mainID])))
                gradSquaredW[mainID][a] += gradMain[a] ** 2

            for a in range(d):
                w2[a] -= ((gradContext[a] * learningRate) / math.sqrt(sum(gradSquaredW[contextID + len(vocab)])))
                gradSquaredW[contextID + len(vocab)][a] += gradContext[a] ** 2

            biases[mainID] -= ((learningRate * gradBiasMain) / math.sqrt(gradSquaredBiases[mainID]))
            biases[contextID + len(vocab)] -= (
                        (learningRate * gradBiasContext) / math.sqrt(gradSquaredBiases[contextID + len(vocab)]))

            gradSquaredBiases[mainID] += gradBiasMain ** 2
            gradSquaredBiases[contextID + len(vocab)] += gradBiasContext ** 2

            W[mainID] = w1
            W[contextID + len(vocab)] = w2

        print ("For Iteration = " + str(i) + " Global cost = " + str(globalCost))

    return W


def writeOutVectors(Y, vectorSize):
    fout = open("C:\\Users\\atifh\\Desktop\\GloVe\\out.txt", "w")
    for row in Y:
        string = ""
        for i in range(vectorSize):
            string += str(row[i]) + ","
        fout.write(string + "\n")
    fout.close()


vocab, corpus = buildCorpusAndVocab()
print("vocab and corpus built")

windowSize, iterations, vectorSize, learningRate, xmax, alpha = 10, 25, 100, 0.05, 100, 0.75

X = buildCoocM(vocab, corpus, windowSize)
print("Matrix built")

W = gloveTrain(vocab, X, iterations, vectorSize, learningRate, xmax, alpha)
Y = list()

# Summation of the vector of a word occurring as main and context
for i in range(len(vocab)):
    Y.append([(W[i][a] + W[i + len(vocab)][a]) for a in xrange(vectorSize)])
print("\n\n\n")

writeOutVectors(Y, vectorSize)

# Find the 30 most similar words to the given word
word = "military"
index = vocab.index(word)
print(vocab[index] + " : ")

dists = np.dot(np.array(Y), np.array(Y[index]))
z = list()
for i in range(len(vocab)):
    z.append([dists[i], i])
z = sorted(z, key=lambda x: x[0], reverse=True)
for i in range(30):
    print(vocab[z[i][1]])







