# coding: utf-8

__author__ = 'Rudakov Kirill'

# IMPORT
import pymorphy2
import codecs
import pickle as p
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier as mc
from sklearn import metrics
from sklearn import linear_model
import sys

#
file=open('dump','rb')
dump = p.load(file)
predicted = dump.toarray()
file.close()
# print (dump)

#
file=open('prob','rb')
prob = p.load(file)
file.close()
# print prob

# Как получить 2510 и 88??

def classifyAndCount(predicted, probability = []):
    quantity = [0.]*88
    probQuantity = [0.]*88
    if len(probability) == 0:
        print 'Classify And Count:'
        for i in range(2510):
            for j in range(88):
                if predicted[i][j] == 1:
                    quantity[j] += 1
        for j in range(88):
            quantity[j] = quantity[j]/2510
        print quantity
    else:
        print 'Probabilistic Classify And Count :'
        for i in range(2510):
            for j in range(88):
                probQuantity[j] += probability[i][j]
        for j in range(88):
                probQuantity[j] = probQuantity[j]/2510
        print probQuantity


classifyAndCount(predicted)
classifyAndCount(predicted,prob)

quit()



# for all indump:
#     print type(SVC.predict_proba(all[0][0]))