#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import jieba
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

# 结巴分词处理，保存处理后的数据
def jiebacut():
    negdataMat = []
    posdataMat = []

    negfile = open('../PythonScraping/neg.txt')
    posfile = open('../PythonScraping/pos.txt')
    for line in negfile.readlines():
        wordline = list(jieba.cut(line.strip()))
        negdataMat.append(wordline)
    for line in posfile.readlines():
        wordline = list(jieba.cut(line.strip()))
        posdataMat.append(wordline)
    m1 = len(posdataMat)
    m2 = len(negdataMat)
    print m1
    print m2
    np.save('./data/neg_jieba',negdataMat)
    np.save('./data/pos_jieba',posdataMat)
# 用jieba分词得到的语料库喂给word2vec进行学习训练
def w2vmodel():
    neg = np.load('./data/neg_jieba.npy')
    pos = np.load('./data/pos_jieba.npy')
    m1 = len(list(pos)); m2 = len(list(neg))
    print m1,m2
    corpus = np.concatenate((neg,pos))
    print corpus.shape
    w2v_model = Word2Vec(corpus, size = 128, min_count = 5, workers = 4)
    w2v_model.save('./w2v_model')


# 最简单粗暴的方法，将每句话每个词编码得到向量然后加和取平均
# 我们这里要得到一行数据用于svm分类。
def build_sentence_vec(text, size, w2vmodel):
    data = np.zeros((1,size))
    count = 0
    for word in text:
        try:
            data += w2vmodel[word].reshape(1,size)
            count += 1
        except KeyError:
            continue
    if count != 0:
        data /= count
    return data

def build_data():
    imdb_w2v = Word2Vec.load('./w2v_model')
    neg = np.load('./data/neg_jieba.npy')
    pos = np.load('./data/pos_jieba.npy')
    m1 = len(list(neg))
    m2 = len(list(pos))
    y = np.concatenate((np.ones((m2,1)),np.zeros((m1,1))))
    x_train,x_test,y_train,y_test = train_test_split(np.concatenate((pos,neg)), y, test_size = 0.2)

    size = 128
    train_vec = np.concatenate([build_sentence_vec(sents, size, imdb_w2v) for sents in x_train])
    test_vec = np.concatenate([build_sentence_vec(sents, size, imdb_w2v) for sents in x_test])
    print train_vec.shape
    print test_vec.shape
    np.save('./data/train_vec', train_vec)
    np.save('./data/test_vec', test_vec)
    np.save('./data/y_train', y_train)
    np.save('./data/y_test', y_test)

# svm的模型。
def svm_model():
    train_vec = np.load('./data/train_vec.npy')
    test_vec = np.load('./data/test_vec.npy')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')

    clf = SVC(kernel = 'rbf', verbose = True)
    clf.fit(train_vec, y_train)
    joblib.dump(clf, './svc_model')
    print 'score:',clf.score(test_vec, y_test)

def svm_test():
    clf = joblib.load('./svc_model')
    pretext = '这家店装修'
    vec = build_sentence_vec(list(jieba.cut(pretext)),128,Word2Vec.load('./w2v_model'))
    result = clf.predict(vec)
    if result == 1:
        print "好评"
    if result == 0:
        print '差评'

# 考虑cnn+lstm的模型。
# 首先看一下每句话的长度分布，因为要设定一个统一的矩阵，矩阵的列数为单词的个数
def sents_length():
    neg = np.load('./data/neg_jieba.npy')
    pos = np.load('./data/pos_jieba.npy')
    data = np.concatenate((neg,pos))
    length = pd.Series(([len(sents) for sents in data]))
    # pandas的quantile可以求出数据的上下4分位数，这里百分之70大于，80大于
    # 76 99 146
    print length.quantile([.7,.8,.9])
def rnn_build_sentense(text, size, w2vmodel, pading_size):
    sents = []
    for item in range(pading_size):
        try:
            sents_mat = w2vmodel[text[item]]
        except:
            #分两种情况，长度不够或者w2vmodel中没有这个词
            sents_mat = ([0] * size)
        sents.append(sents_mat)
    return sents
def rnn_build_data():
    imdb_w2v = Word2Vec.load('./w2v_model')
    neg = np.load('./data/neg_jieba.npy')
    pos = np.load('./data/pos_jieba.npy')
    m1 = len(list(neg))
    m2 = len(list(pos))
    y = np.concatenate((np.ones((m2,1)),np.zeros((m1,1))))
    x_train,x_test,y_train,y_test = train_test_split(np.concatenate((pos,neg)), y, test_size = 0.2)
    size = 128
    pading_size = 100
    test_vec = [];train_vec = [];i = 0
    print x_test.shape
    for sents in x_test:
        test_vec.append(rnn_build_sentense(sents,size,imdb_w2v,pading_size))
        print i
        i += 1
    for sents in x_train:
        train_vec.append(rnn_build_sentense(sents,size,imdb_w2v,pading_size))
        print i
        i += 1

    # 64行指的是每句话的取出64个单词每个单词编码成128列向量
    test_vec = np.reshape((test_vec),(-1,pading_size,128))
    test_vec = np.reshape((test_vec),(-1,1,pading_size,128))
    train_vec = np.reshape((train_vec),(-1,pading_size,128))
    train_vec = np.reshape((train_vec),(-1,1,pading_size,128))
    print test_vec.shape
    np.save('./data/rnn_small_xtest_vec', test_vec)
    np.save('./data/rnn_y_small_test', y_test)
    np.save('./data/rnn_small_xtrain_vec', train_vec)
    np.save('./data/rnn_y_small_train', y_train)


def cluster():
    w2vmodel = Word2Vec.load('./w2v_model')
    word_vector = w2vmodel.syn0
    print word_vector.shape
    wordnumber = word_vector.shape[0]
    num_clusters = 128
    kmeans_clustering = KMeans(n_clusters = num_clusters, n_jobs = 8,verbose = 1)
    idx = kmeans_clustering.fit(word_vector)
    word_centroid_map = dict(zip(w2vmodel.index2word, idx))
    print word_centroid_map
    filename = 'word_centroid_map_10avg.pickle'
    pickle.dump(word_centroid_map, filename)
cluster()