# coding: utf-8
__author__ = 'Nikolay Karpov'

from sklearn.cross_validation import KFold, StratifiedKFold, train_test_split
from quantification import Quantification
from sklearn.datasets import make_classification
import numpy as np
from SVMperfRealization import SVMperf
from sklearn.datasets import fetch_20newsgroups
import scipy
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from numpy import concatenate
import pandas as pd
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC
from sklearn import linear_model
from text_processing import  text_processing, char_processing
from semeval2016 import *

def generate_data(n_samples=1000):
    #X, y, p_c, p_w_c = make_ml_data(n_samples=1000, n_features=10,n_classes=3, n_labels=1,length=50, allow_unlabeled=False, return_indicator=True,return_distributions=True, sparse=True)
    X,y=make_classification(n_samples=n_samples, n_features=10, n_informative=3, n_classes=2, n_clusters_per_class=2)
    #index=[]
    #for key in range(len(scipy.unique(y))):
    #    index.append(key)
    #y=MultiLabelBinarizer(classes=index).fit_transform([[y_p] for y_p in y])

    #kf=KFold(y.shape[0],n_folds=2)
    #for train_index, test_index in kf:
    #    q.fit(X[train_index],y[train_index])
        #prev_pred=q.predict(X[test_index])
        #print('prev_pred',prev_pred)
    #print(y)
    return X, y

def quantify_OHSUMED():
    q=Quantification(method='test', dir_name='QuantOHSUMED') #QuantRCV1 #QuantOHSUMED
    [y_train,y_test_list,pred_prob_list,test_files, y_names]=q._read_pickle('texts/cl_prob_'+q.dir_name+'.pickle')
    indexes=q._read_pickle('texts/cl_indexes_'+q.dir_name+'.pickle')
    td=q._classify_and_count(y_test_list)
    #ed=q._expectation_maximization(y_train, pred_prob_list, stop_delta=0.1)
    ed=q._exp_max(y_train, pred_prob_list, stop_delta=0.1)
    #ed=q._prob_classify_and_count(pred_prob_list)
    r=q._count_diff(td, ed)
    return r

def gen_texts():
    categories = ['alt.atheism','talk.religion.misc','comp.graphics', 'sci.space']
    dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
    labels = dataset.target
    return dataset.data, labels


def quantify(X_train, X_test, y_train, y_test,method):
    q=Quantification(method=method)
    q.fit(X_train,y_train)
    KLD=q.score([X_test], [y_test])
    #print('Test',q._classify_and_count([y_test]))
    #print('Train',q._classify_and_count([y_train]))
    return KLD

def test_gen_data():
    k_CC=0
    k_PCC=0
    k_ACC=0
    k_PACC=0
    k_EM=0
    k_SVMp=0
    k_Iter=0
    k_EM1=0
    CC=[]
    PCC=[]
    ACC=[]
    PACC=[]
    EM=[]
    EM1=[]
    SVMp=[]
    Iter=[]
    Iter1=[]
    for i in range(1):
        X,y=generate_data(n_samples=10000)
        X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.75)
        q=Quantification(is_clean=True)
        X_test_d,y_test_d=Quantification.make_drift_list(X_test,y_test,0.75)
        q.fit(X_train,y_train)
        #s = SVMperf(X_train, y_train, X_test_d, y_test_d)
        #SVMp.append(q._kld(q._classify_and_count([y_test]), q._classify_and_count([s.getPredictions()])))
        CC.append(q.score(X_test_d, y_test_d, method='CC'))
        PCC.append(q.score(X_test_d, y_test_d, method='PCC'))
        ACC.append(q.score(X_test_d, y_test_d, method='ACC'))
        PACC.append(q.score(X_test_d, y_test_d, method='PACC'))
        EM1.append(q.score(X_test_d, y_test_d, method='EM1'))
        EM.append(q.score(X_test_d, y_test_d, method='EM'))
        Iter.append(q.score(X_test_d, y_test_d, method='Iter'))
        Iter1.append(q.score(X_test_d, y_test_d, method='Iter1'))

        #print(i, q._classify_and_count([y_test]))

        if Iter1[i]<=PCC[i]: k_PCC+=1
        if Iter1[i]<=CC[i]: k_CC+=1
        if Iter1[i]<=ACC[i]: k_ACC+=1
        if Iter1[i]<=PACC[i]: k_PACC+=1
        if Iter1[i]<=EM[i]: k_EM+=1
        #if EM1[i]<=SVMp[i]: k_SVMp+=1
        if Iter1[i]<=EM1[i]: k_EM1+=1
        if Iter1[i]<=Iter[i]: k_Iter+=1
        #print(q.predict(X_test_d, method='CC'),'\n',q.predict(X_test_d, method='PCC'),'\n',q.predict(X_test_d, method='ACC'),'\n',q.predict(X_test_d, method='PACC'),'\n',q.predict(X_test_d, method='EM1'),'\n',q.predict(X_test_d, method='EM'),'\n',SVMp,'\n',q.predict(X_test_d, method='Iter'))
        print(i, q._classify_and_count(y_test_d),'\n')
    print('CC\t','PCC\t','ACC\t','PACC\t','EM1\t','EM\t','SVMperf\t','Iter\t','Iter1\t')
    print(np.average(CC),'\t',np.average(PCC),'\t',np.average(ACC),'\t',np.average(PACC),'\t',np.average(EM1),'\t',np.average(EM),'\t',np.average(SVMp),'\t',np.average(Iter),'\t',np.average(Iter1),'\t\t\t',
          np.var(CC),'\t',np.var(PCC),'\t',np.var(ACC),'\t',np.var(PACC),'\t',np.var(EM1),'\t',np.var(EM),'\t',np.var(SVMp),'\t',np.var(Iter),'\t',np.var(Iter1),'\t\t\t',
          k_CC,'\t',k_PCC,'\t',k_ACC,'\t',k_PACC,'\t',k_EM1,'\t',k_EM,'\t',k_SVMp,'\t',k_Iter)


from SVMperfRealization import *
def forSVMperf(q,y_train,topics,y_test_list,utopics):
    pred_prev=getOutData(y_train,topics)

    dist_list=[]
    real_prev=[]
    for y_test in y_test_list:
        real_prev.append(q._classify_and_count(y_test))
    for i,topic in enumerate(utopics):
        print(real_prev[i],pred_prev[topic])
        dist_list.append(q._emd(real_prev[i],pred_prev[topic]))
    print(np.average(dist_list))

def predict_set(q, X_list, y_list, method=''):
    scores=[]
    ind_pred=[]
    ind_real=[]
    for i,X in enumerate(X_list):
        prev_pred=q.predict(X,method)
        scores.append(prev_pred)
        for item in q.iter_model.predict(X):
            ind_pred.append(item)
        for item in y_list[i]:
            ind_real.append(item)
    res=metrics.classification_report(ind_real,ind_pred)
    print(res)
    return scores

def semEval():
    #fname='100_topics_100_tweets.sentence-three-point.subtask-A'
    fname='100_topics_100_tweets.topic-five-point.subtask-CE'
    #fname='100_topics_XXX_tweets.topic-two-point.subtask-BD'

    #train=read_semeval('texts/2download/gold/all/'+fname+'.all.gold.tsv')
    train=read_semeval('texts/2download/gold/train/'+fname+'.train.gold.tsv')
    tp=text_processing(n=4)
    cp=char_processing()
    Xw_train=tp.fit_transform(train[0])
    #pos_score_train, neg_score_train=readSentiScores('texts/2download/train+.out')
    #add1=csr_matrix(pos_score_train).transpose()
    #add2=csr_matrix(neg_score_train).transpose()
    #Xw_train=scipy.sparse.hstack((scipy.sparse.hstack((Xw_train,add1)),add2), format='csr')

    Xc_train=cp.fit_transform(train[0])
    X_train=scipy.sparse.hstack((Xw_train, Xc_train), format='csr')
    #X_train=Xw_train
    y_train=np.asarray(train[1])

    #test=read_semeval('texts/2download/test_datasets-v2.0/SemEval2016-task4-test.subtask-BD.txt')
    test=read_semeval('texts/2download/gold/devtest/'+fname+'.devtest.gold.tsv')
    #test=read_semeval('texts/2download/gold/dev/'+fname+'.dev.gold.tsv')
    Xw_test=tp.transform(test[0])
    #pos_score_test, neg_score_test=readSentiScores('texts/2download/test+.out')
    #add1=csr_matrix(pos_score_test).transpose()
    #add2=csr_matrix(neg_score_test).transpose()
    #Xw_test=scipy.sparse.hstack((scipy.sparse.hstack((Xw_test,add1)),add2), format='csr')

    Xc_test=cp.transform(test[0])
    X_test=scipy.sparse.hstack((Xw_test, Xc_test), format='csr')
    #X_test=Xw_test
    y_test=np.asarray(test[1])

    #perf=SVMperf(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
    #X_train, X_test, y_train, y_test=train_test_split(X_train,y_train, test_size=0.75)
    print(X_test.shape)
    X_test_list, y_test_list, utopics=split_by_topic(X_test, y_test, test[2])

    q=Quantification(method='Iter1',is_clean=True)
    #X_test, y_test=Quantification.make_drift_list(X_test.toarray(), y_test, proportion=0.2)
    q.fit(X_train, y_train)
    print('train',q._classify_and_count(y_train))

    prevs=q.predict_set(X_test_list, method='Iter1')
    write_semeval(dict(zip(utopics,prevs)),fname=fname)
    #forSVMperf(q,y_test,test[2],y_test_list, utopics)
    #q.iter_model.predict()

    print('CC',q.score(X_test_list,y_test_list, method='CC'))
    print('PCC',q.score(X_test_list,y_test_list, method='PCC'))
    print('EM',q.score(X_test_list,y_test_list, method='EM'))
    print('EM1',q.score(X_test_list,y_test_list, method='EM1'))
    print('Iter',q.score(X_test_list,y_test_list, method='Iter'))
    print('Iter1',q.score(X_test_list,y_test_list, method='Iter1'))
    print('ACC',q.score(X_test_list,y_test_list, method='ACC'))
    print('PACC',q.score(X_test_list,y_test_list, method='PACC'))

def after_semEval():
    #fname='100_topics_100_tweets.sentence-three-point.subtask-A'
    #fname='100_topics_100_tweets.topic-five-point.subtask-CE'
    fname='100_topics_XXX_tweets.topic-two-point.subtask-BD'

    train=read_semeval('texts/2download/gold/all_train/'+fname+'.all.gold.tsv')
    #train=read_semeval('texts/2download/gold/all/'+fname+'.train.gold.tsv')
    tp=text_processing(n=1)
    #cp=char_processing()
    Xw_train=tp.fit_transform(train[0])
    #pos_score_train, neg_score_train=readSentiScores('texts/2download/train+.out')
    #add1=csr_matrix(pos_score_train).transpose()
    #add2=csr_matrix(neg_score_train).transpose()
    #Xw_train=scipy.sparse.hstack((scipy.sparse.hstack((Xw_train,add1)),add2), format='csr')

    #Xc_train=cp.fit_transform(train[0])
    #X_train=scipy.sparse.hstack((Xw_train, Xc_train), format='csr')
    X_train=Xw_train
    y_train=np.asarray(train[1])

    #test=read_semeval('texts/2download/test_datasets-v2.0/SemEval2016-task4-test.subtask-BD.txt')
    test=read_semeval('texts/2download/gold/test/test_gold_2.csv')
    #test=read_semeval('texts/2download/gold/dev/'+fname+'.dev.gold.tsv')
    Xw_test=tp.transform(test[0])
    #pos_score_test, neg_score_test=readSentiScores('texts/2download/test+.out')
    #add1=csr_matrix(pos_score_test).transpose()
    #add2=csr_matrix(neg_score_test).transpose()
    #Xw_test=scipy.sparse.hstack((scipy.sparse.hstack((Xw_test,add1)),add2), format='csr')

    #Xc_test=cp.transform(test[0])
    #X_test=scipy.sparse.hstack((Xw_test, Xc_test), format='csr')
    X_test=Xw_test
    y_test=np.asarray(test[1])

    #perf=SVMperf(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
    #X_train, X_test, y_train, y_test=train_test_split(X_train,y_train, test_size=0.75)
    print(X_test.shape)
    X_test_list, y_test_list, utopics=split_by_topic(X_test, y_test, test[2])

    q=Quantification(method='Iter1',is_clean=True)
    #X_test, y_test=Quantification.make_drift_list(X_test.toarray(), y_test, proportion=0.2)
    q.fit(X_train, y_train)
    print('train',q._classify_and_count(y_train))

    #prevs=q.predict_set(X_test_list, method='Iter1')
    #write_semeval(dict(zip(utopics,prevs)),fname=fname)
    #forSVMperf(q,y_test,test[2],y_test_list, utopics)
    #q.iter_model.predict()

    print('CC',q.score(X_test_list,y_test_list, method='CC'))
    print('PCC',q.score(X_test_list,y_test_list, method='PCC'))
    print('EM',q.score(X_test_list,y_test_list, method='EM'))
    print('EM1',q.score(X_test_list,y_test_list, method='EM1'))
    print('Iter',q.score(X_test_list,y_test_list, method='Iter'))
    print('Iter1',q.score(X_test_list,y_test_list, method='Iter1'))
    print('ACC',q.score(X_test_list,y_test_list, method='ACC'))
    print('PACC',q.score(X_test_list,y_test_list, method='PACC'))

#quantify_OHSUMED()
#test_gen_data()
after_semEval()
#semEvalP()
#q=Quantification(method='CC')
#print(np.average([q._emd([0.0,0.11,0.5,0.29,0.100],[0.0,0.11,0.5,0.2,0.19]), q._emd([0.2,0.25,0.5,0.05,0.0],   [0.0,0.25,0.4,0.15,0.2])]))
#read_and_swop()