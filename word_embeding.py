# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : word_embeding.py
# Time    : 2018/12/11 0011 下午 3:10
"""


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import pickle
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler


class word_embeding(object):
    def __init__(self, X, n_dim, vocab_path='./', tfidf_path='./', w2v_path='./', ngram=1):
        'X input type = [[a,b,c],[e,f,g]] segment file'
        self.X = X
        self.vocab_path = vocab_path
        self.tfidf_path = tfidf_path
        self.ngram = ngram
        self.n_dim = n_dim
        self.w2v_path = w2v_path

    def tf_idf_fit(self):
        vectorizer = TfidfVectorizer(min_df=10, max_df=0.5, norm='l2', token_pattern=r"(?u)\b\w+\b",
                                     ngram_range=(1, self.ngram), dtype=np.int32)
        x = [' '.join(i) for i in self.X]
        vectorizer.fit(x)
        with open(self.vocab_path, 'w', encoding='utf8') as f:
            for w, i in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
                f.write(w)
                f.write('\t')
                f.write(str(i))
                f.write('\n')
        pickle.dump(vectorizer, open(self.tfidf_path, 'wb'))

    def tf_idf_transform_vector(self):
        x1 = [' '.join(i) for i in self.X]
        tf_model = pickle.load(open(self.tfidf_path, 'rb'))
        x1 = tf_model.transform(x1)
        x1 = csr_matrix(x1)
        return x1

    def word2vec_fit(self):
        # 初始化模型建立vocab词向量
        w2v = Word2Vec(size=self.n_dim, min_count=5)
        w2v.build_vocab(self.X)
        print(w2v.corpus_count)

        # 训练模型
        w2v.train(self.X, total_examples=w2v.corpus_count, epochs=w2v.iter)
        w2v.save(self.w2v_path)

    def word2vec_transform_vector(self):
        w2v = Word2Vec.load(self.w2v_path)
        x_train_vec = np.concatenate([self.buid_word_vector(z, self.n_dim, w2v) for z in self.X])
        x_train_vec = MinMaxScaler(x_train_vec)
        return x_train_vec

    def buid_word_vector(self, text, size, w2v):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in text:
            try:
                vec += w2v[word].reshape((1, size))
                count += 1
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec


