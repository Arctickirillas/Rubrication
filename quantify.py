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
#file=open('dump','rb')
# dump = p.load(file)
# predicted = dump.toarray()
# file.close()
# # print (dump)
#
# #
# file=open('prob','rb')
# prob = p.load(file)
# file.close()
# print prob

# Как получить 2510 и 88??

def classifyAndCount(predicted, probability = []):
    quantity = [0.]*88
    probQuantity = [0.]*88
    if len(probability) == 0:
        print('Classify And Count:')
        for i in range(2510):
            for j in range(88):
                if predicted[i][j] == 1:
                    quantity[j] += 1
        for j in range(88):
            quantity[j] = quantity[j]/2510
        #print quantity
    else:
        #print 'Probabilistic Classify And Count :'
        for i in range(2510):
            for j in range(88):
                probQuantity[j] += probability[i][j]
        for j in range(88):
                probQuantity[j] = probQuantity[j]/2510
        #print probQuantity


def CCforDouble(predicted, probability = []):
    quantityNeg, quantityPos = 0.,0.
    probQuantityNeg = 0.0
    probQuantityPos = 0.0
    array = [0.0]*2
    if len(probability) == 0:
        # print 'Classify And Count:'
        for i in range(len(predicted)):
            if str(predicted[i]) == 'negative' or str(predicted[i]) == '-1' or str(predicted[i]) == '0':
                quantityNeg += 1
            elif str(predicted[i]) == 'positive' or str(predicted[i]) == '1':
                quantityPos += 1
        quantityNeg = quantityNeg/len(predicted)
        quantityPos = quantityPos/len(predicted)
        return quantityPos,quantityNeg
    else:
        # print 'Probabilistic Classify And Count :'
        probability = np.asarray(probability).transpose()
        for j in range(len(probability)):
            array[j] = np.average(probability[j])
        return array[1],array[0]


# classifyAndCount(predicted)
# classifyAndCount(predicted,prob)
#
# quit()



# for all indump:
#     print type(SVC.predict_proba(all[0][0]))