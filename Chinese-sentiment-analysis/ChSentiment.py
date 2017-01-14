#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import jieba
from gensim.models.word2vec import Word2Vec
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split

# 载入数据，分词
def load_file_and_preprocessing():
  neg = pd.read_excel('./data/neg.xls', header=None, index = None)
  pos = pd.read_excel('./data/pos.xls', header=None,  index = None)
  m1, n1 = neg.shape
  m2, n2 = pos.shape
  print np.ones((m1))
  y = np.concatenate((np.ones((m2,1)), np.zeros((m1,1))))
  cw = lambda x: list(jieba.cut(x))
  pos['words'] = pos[0].apply(cw)
  neg['words'] = neg[0].apply(cw)
  x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size = 0.2)
  np.save('./y_train',y_train)
  np.save('./y_test',y_test)
  np.save('./x_train',x_train)
  np.save('./x_test',x_test)
  return x_train, x_test

# 对每个句子的所有词向量取平均
def build_sentense_vector(text, size, imdb_w2v):
  vec = np.zeros((1,size))
  count = 0.0
  for word in text:
    try:
      vec += imdb_w2v[word].reshape((1,size))
    except KeyError:
      continue
  if count != 0:
    vec /= count
  return vec

# 计算得到词向量
def get_train_vecs(x_train, x_test):
  n_dim = 300
  # 初始化word2Vec模型:
  imdb_w2v = Word2Vec(size = n_dim, min_count = 10)
  imdb_w2v.build_vocab(x_train)

  imdb_w2v.train(x_train)
  train_vec = np.concatenate([build_sentense_vector(word, n_dim, imdb_w2v) for word in x_train])
  imdb_w2v.train(x_test)
  imdb_w2v.save('./w2v_model')
  test_vec = np.concatenate([build_sentense_vector(word, n_dim, imdb_w2v) for word in x_test])
  np.save('./train_vec', train_vec)
  np.save('./test_vec', test_vec)
  print 'train_vec.shape:', train_vec.shape
  print 'test_vec.shape:', test_vec.shape

def svm_train():
  train_vec = np.load('./train_vec.npy')
  test_vec = np.load('./test_vec.npy')
  y_train = np.load('./y_train.npy')
  y_test = np.load('./y_test.npy')
  clf = SVC(kernel = 'rbf', verbose = True)
  clf.fit(train_vec, y_train)
  joblib.dump(clf, './svc_model')
  print 'score:', clf.score(test_vec, y_test)

def svm_predict(string1):
  words = jieba.cut(string1)
  n_dim = 300
  imdb_w2v = Word2Vec.load('./w2v_model')
  vect = build_sentense_vector(words, n_dim, imdb_w2v)
  clf = joblib.load('./svc_model')
  result = clf.predict(vect)
  if int(result) == 1:
    print 'positive'
  if int(result) == 0:
    print 'negitave'


# load_file_and_preprocessing()
# x_train = np.load('./x_train.npy')
# x_test = np.load('./x_test.npy')
# get_train_vecs(x_train, x_test)
# svm_train()
x_train = np.load('./x_train.npy')
print x_train
svm_predict('台灯质量一般，灯太暗，还有卖家及其不讲信用')












