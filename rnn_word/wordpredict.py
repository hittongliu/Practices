#-*- coding: utf-8 -*-

import numpy as np
import nltk
import os
import sys
from gensim.models.word2vec import Word2Vec


reload(sys)
sys.setdefaultencoding('utf8')
raw_text = ''
for file in os.listdir("./input/"):
    if file.endswith('.txt'):
        raw_text += open("./input/" + file).read() + '\n\n'
raw_text = raw_text.lower()
sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
sents = sentensor.tokenize(raw_text)
corpus = []
for sen in sents:
    corpus.append(nltk.word_tokenize(sen))
print (len(corpus))
print (corpus[:3])

w2v_model = Word2Vec(corpus, size = 128, window = 5, min_count = 5, workers = 4)

print w2v_model['office']
raw_input = [item for sublist in corpus for item in sublist]
test_stream = []
vocab = w2v_model.vocab
for word in raw_input:
    if word in vocab:
        test_stream.append(word)
seq_lenth = 10
x = []
y = []
for i in range(0, len(test_stream) - seq_lenth):
    given = test_stream[i:i+seq_lenth]
    predict = test_stream[i + seq_lenth]
    print np.array([w2v_model[word] for word in given]).shape
    x.append(np.array([w2v_model[word] for word in given]))
    y.append(w2v_model[predict])

x = np.reshape(x, (-1, seq_lenth, 128))
y = np.reshape(y, (-1,128))

# LSTM model
model = Sequential()
model.add(LSTM(256, dropout_W = 0.2, dropout_U = 0.2, input_shape = (seq_lenth, 128)))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'sigmoid'))
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y, nb_epoch = 50, batch_size = 4096)
print "hi"