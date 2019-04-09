# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2018/12/24 0024 上午 10:20
"""


from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer, one_hot
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import nltk
import numpy as np
import operator

np.random.seed(2018)

BATCH_SIZE = 128
NUM_EPOCHS = 20

lines = []
fin = open("alice.txt", "r")
for line in fin:
    line = line.strip()
    if len(line) == 0:
        continue
    lines.append(line)
fin.close()

sents = nltk.sent_tokenize(" ".join(lines)) # 以句子为单位进行划分

tokenizer = Tokenizer(500)  # use top 5000 words only
tokenizer.fit_on_texts(sents)
vocab_size = len(tokenizer.word_counts) + 1
sequences = tokenizer.texts_to_sequences(sents)
print(sequences)



'''
    对每个句子提取出3个连续单词的tuple=(left,center,right)，skipgram模型（假设词窗大小为3）的
    目标是从center预测出left、从center预测出right。
    因此对于每个tuple=(left,center,right)的数据，整理出的两组数据，如[x,y] = [[x1,y1],[x2,y2]]=[ [center,left],[center,right] ] 

'''


xs = []
ys = []
for sequence in sequences[:1]:
    triples = list(nltk.trigrams(sequence)) # 该句子数字序列中，每3个连续的数字组成一个tuple并返回
    w_lefts = [x[0] for x in triples]
    w_centers = [x[1] for x in triples]
    w_rights = [x[2] for x in triples]
    xs.extend(w_centers)
    ys.extend(w_lefts)
    xs.extend(w_centers)
    ys.extend(w_rights)

print(xs)
print(ys)


# 将上面已经得到xs,ys转化为 one-hot矩阵
'''
    例如词典大小为 5，有两个待转化为One-hot编码的数字[[2],[4]],则one-hot编码返回一个矩阵为
    [
      [0,0,1,0,0],
      [0,0,0,0,1]
    ]
'''

ohe = OneHotEncoder(n_values=vocab_size)
X = ohe.fit_transform(np.array(xs).reshape(-1, 1)).todense()
Y = ohe.fit_transform(np.array(ys).reshape(-1, 1)).todense()

# 划分出30%作为测试集，70%作为训练集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3,
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)



model = Sequential()
model.add(Dense(300, input_shape=(Xtrain.shape[1],)))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(Ytrain.shape[1]))
model.add(Activation("softmax"))

model.compile(optimizer="Nadam", loss="categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,
                    epochs=15, verbose=1,
                    validation_data=(Xtest, Ytest))


# plot loss function
plt.subplot(211)
plt.title("accuracy")
plt.plot(history.history["acc"], color="r", label="train")
plt.plot(history.history["val_acc"], color="b", label="validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

# evaluate model
score = model.evaluate(Xtest, Ytest, verbose=1)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score[0], score[1]))


# using the word2vec model
word2idx = tokenizer.word_index
idx2word = {v:k for k, v in word2idx.items()}

# retrieve the weights from the first dense layer. This will convert
# the input vector from a one-hot sum of two words to a dense 300
# dimensional representation
W, b = model.layers[0].get_weights()

# 计算词典所有单词的词向量
idx2emb = {}
for word in word2idx.keys():
    wid = word2idx[word]
    a = np.zeros(526)
    a[wid] = 1
    a = a.reshape(1, 526)
    vec_in = a
    vec_emb = np.dot(vec_in, W)
    idx2emb[wid] = vec_emb
print(word2idx.keys())
# 找出与word的词向量余弦相似度最高的10个单词，并输出这些单词
for word in ['alice', 'her', 'very']:
    wid = word2idx[word]
    source_emb = idx2emb[wid]
    distances = []
    for i in range(1, vocab_size):
        if i == wid:
            continue
        target_emb = idx2emb[i]
        distances.append(
            ((wid, i),
             cosine_distances(source_emb, target_emb)
            )
        )
    sorted_distances = sorted(distances, key=operator.itemgetter(1))[0:10]
    predictions = [idx2word[x[0][1]] for x in sorted_distances]
    print("{:s} => {:s}".format(word, ", ".join(predictions)))




