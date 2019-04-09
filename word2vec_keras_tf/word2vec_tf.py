# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/4/8 0008 上午 10:23
"""

from word_segment import read_file_seg,word_frequence,word_index_word,corups_index,split_data
import tensorflow as tf
import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


'step1 : read corpus'
path = './corpus/test.txt'
vocab, sentence = read_file_seg(path, flag=True)

'step2 :  statistic word frequency and sentence convert to index '

word_fre, vocab_size = word_frequence(vocab)
word2id, id2word = word_index_word(word_fre)


data = corups_index(sentence, word2id)

'step3: split train, test'
X_train, Y_train = split_data(data, win=3)



'step 4 : word2vec skip-gram with negative sample '

class w2v_sk():
    def __init__(self, vocab_size, embedding_size, batch_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.input_x = tf.placeholder(tf.int32, shape=[None], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, 1], name='input_y')
        self.num_negative_samples = 128
        self.optimizer, self.loss, self.normalized_embeddings = self.inference()

    def inference(self):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, self.embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self.input_x)

        nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=1.0 / np.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))

        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=self.input_y, inputs=embed, num_sampled=self.num_negative_samples,
                           num_classes=vocab_size))

        optimizer = tf.train.AdamOptimizer().minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        return optimizer, loss, normalized_embeddings


def train_w2v():
    epoch = 5
    batch_size = 10
    with tf.Session() as sess:
        model = w2v_sk(vocab_size=vocab_size, embedding_size=100, batch_size=batch_size)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, '.model/tf_100')
        for i in range(epoch):
            for j in range(int(X_train.shape[0]/batch_size)):
                s = np.random.randint(0, 848, 1)[0]
                x_batch, y_batch = X_train[s:s + 10], Y_train[s:s + 10]
                _, loss_, normalized_embeddings = sess.run([model.optimizer, model.loss, model.normalized_embeddings],
                                                          feed_dict={model.input_x: x_batch, model.input_y: y_batch})
                print('Batch %d -- Loss %f' % (j, loss_))

            final_embeddings = normalized_embeddings
            with open('./model/tf_128.pkl', 'wb') as fw:
                pickle.dump({'embeddings': final_embeddings, 'word2id': word2id, 'id2word': id2word}, fw, protocol=4)




train_w2v()


def show_word():
    with open('tf_100.pkl', 'rb') as fr:
        data = pickle.load(fr)
        final_embeddings = data['embeddings']
        word2id = data['word2id']
        id2word = data['id2word']

    word_indexs = []
    count = 0
    plot_only = 200
    for i in range(1, len(id2word.keys())):
        word_indexs.append(i)
        count += 1
        if count == plot_only:
            break

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[word_indexs, :])
    labels = [id2word[str(i)] for i in word_indexs]

    plt.figure(figsize=(15, 12))
    for i, label in enumerate(labels):
        x, y = two_d_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, (x, y), ha='center', va='top', fontproperties='Microsoft YaHei')
    plt.show()
