#-*- coding: utf-8 -*-

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import sys
reload(sys)
sys.setdefaultencoding('utf8')


def ft_build_data():
    neg = np.load('./data/neg_jieba.npy')
    pos = np.load('./data/pos_jieba.npy')
    m1 = len(list(neg))
    m2 = len(list(pos))
    y = np.concatenate((np.ones((m2,1)),np.zeros((m1,1))))
    x_train,x_test,y_train,y_test = train_test_split(np.concatenate((pos,neg)), y, test_size = 0.2)
    print y_train.shape
    for i in range(len(y_train)):
        label = '__label__' + str(y_train[i][0])
        x_train[i].append(label)

    x_train = [' '.join(x) for x in x_train]

    x_test = [' '.join(x) for x in x_test]
    print x_test[10]

    file1 = open('./data/train_ft','a')
    file2 = open('./data/test_ft','a')
    for sent in x_train:
        file1.write(sent)
        file1.write('\n')
    for sent in x_test:
        file2.write(sent)
        file2.write('\n')
    file1.close()
    file2.close()
    np.save('./data/ylabel_ft',y_test)

def ft_scores():
    file = open('./fastText/predict.txt')
    labels = []
    for line in file.readlines():
        data = line.strip()
        if data == '__label__0.0':labels.append(0)
        if data == '__label__1.0':labels.append(1)
    print len(labels)

    y_preds = np.array(labels).flatten().astype(int)
    print y_preds.shape


    y_test = np.load('./data/ylabel_ft.npy')
    print y_test.shape
    from sklearn import metrics

    # 算个AUC准确率
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_preds, pos_label=1)
    print(metrics.auc(fpr, tpr))
ft_scores()


