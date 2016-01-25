# coding: utf-8
__author__ = 'Nikolay Karpov'

from parse_arff import Parse_ARFF
import pickle
import numpy as np
import operator
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import normalize, Normalizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier as mc
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn import linear_model
from sklearn.mixture import GMM, VBGMM
import os
import scipy
import random

class Quantification:
    def __classificator(self, class_weight='auto'):
        if class_weight=='':
            return SVC(kernel='rbf', probability=True)
            #return linear_model.LogisticRegression()
            #return GMM(n_components=2)
        else:
            return SVC(kernel='rbf', probability=True, class_weight = class_weight)
            #return linear_model.LogisticRegression(class_weight=class_weight)
            #return GMM(n_components=2)

    def __init__(self, method='', dir_name='temp', is_clean=True):
        self.prefix='texts/'
        self.arff=Parse_ARFF()
        self.dir_name=dir_name
        if is_clean: self.__clean_dir(self.prefix+self.dir_name)
        self.n_folds=5
        self.classes=[0,1]
        self.method_prev=self._bin_prevalence#._bin_prevalence or ._multi_prevalence
        self.model=self.__classificator(class_weight='auto')
        if method=='EM' or method=='EM1' or method=='Iter' or method=='Iter1':
            self.method=method
        elif method=='PCC' or method=='CC' or method=='ACC' or method=='PACC':
            self.method=method
        elif method=='test':
            self.method=method
            self._train_file, self._test_files=self.arff.read_dir(self.prefix+'pickle_'+dir_name)
        elif method=='':
            self.method='CC'

    def fit(self, X, y):
        if isinstance(y, list): y=np.asarray(y)
        self.classes=np.unique(y)
        #if isinstance(y, csr_matrix):
        #    self.y_train=y.toarray()
        #elif isinstance(y, np.ndarray):
        #    if len(y.shape)==1:
        #        self.y_train=MultiLabelBinarizer(classes=self.classes).fit_transform([[y_p] for y_p in y])
        #    elif len(y.shape)==2:
        #        self.y_train=y
        self.y_train=y
        self.X_train=X
        self.model.fit(X, y)
        return self.model

    def predict(self, X, method=''):
        if method!='':
            self.method=method
        if self.method=='CC':
            y_pred=self.model.predict(X)
            #print('CC', y_pred)
            prevalence=self._classify_and_count(y_pred)
        elif self.method=='ACC':
            y_pred=self.model.predict(X)
            self.kfold_results=self.__kfold_tp_fp(self.X_train, self.y_train, n_folds=self.n_folds)
            prevalence=self._adj_classify_and_count(y_pred, is_prob=False)
        elif self.method=='PCC':
            prob_pred=self.model.predict_proba(X)
            prevalence=self._prob_classify_and_count(prob_pred)
        elif self.method=='PACC':
            self.kfold_results=self.__kfold_prob_tp_fp(self.X_train, self.y_train, n_folds=self.n_folds)
            prob_pred=self.model.predict_proba(X)
            prevalence=self._adj_classify_and_count(prob_pred, is_prob=True)
        elif self.method=='EM':
            prob_pred=self.model.predict_proba(X)
            prevalence=self._expectation_maximization(self.y_train, prob_pred, stop_delta=0.00001)
        elif self.method=='EM1':
            prob_pred=self.model.predict_proba(X)
            prevalence=self._exp_max(self.y_train, prob_pred, stop_delta=0.00001)
        elif self.method=='Iter':
            prevalence=self._cost_sens_learning(X, stop_delta=0.00001, class_weight_start='auto')
        elif self.method=='Iter1':
            prevalence=self._cost_sens_learning(X, stop_delta=0.00001, class_weight_start='')
        elif self.method=='test':
            self._process_pipeline()
        return prevalence

    def predict_set(self, X_list, method=''):
        scores=[]
        for X in X_list:
            prev_pred=self.predict(X,method)
            scores.append(prev_pred)
        return scores

    def score(self, X_list, y_list, method=''):
        scores=[]
        for X, y in zip(X_list, y_list):
            y=np.asarray(y)
            prev_pred=self.predict(X,method)
            prev_true=self._classify_and_count(y)
            #print(prev_pred, prev_true)
            #scores.append(self._divergence_bin(prev_true, prev_pred, self._kld))
            scores.append(self._emd(prev_true, prev_pred))
        return np.average(scores)

    def make_drift_rnd(X,y,proportion=0.5):
        index={}
        for val in scipy.unique(y):
            index[val]=[]
        for key in range(len(y)):
            index[y[key]].append(key)
        ind2low=[]
        num2low=int(len(index)/2)
        while ind2low==[] and num2low!=0:
            j=0
            for i in index:
                #print(i, j, num2low)
                if j>=num2low:
                    break
                rnd=random.random()
                #print(rnd,j,i)
                if rnd<0.5:
                    ind2low.append(i)
                    j+=1
        new_ind=index.copy()
        new_set=[]
        for ind in ind2low:
            for val in index[ind]:
                rnd=random.random()
                if rnd > proportion:
                    new_set.append(val)
            new_ind[ind]=new_set
        new_y=[]
        new_X=[]

        for i in index:
            try:
                new_y=np.concatenate((new_y,y[new_ind[i]]))
                new_X=np.concatenate((new_X,X[new_ind[i]]),axis=0)
            except:
                new_y=y[new_ind[i]]
                new_X=X[new_ind[i]]
        return new_X, new_y

    def make_drift_05(X,y,proportion=0.5):
        #index=[]
        #for key in range(len(scipy.unique(y))):
        #    index.append(key)
        #y=MultiLabelBinarizer(classes=index).fit_transform([[y_p] for y_p in y])
        ind2low=[]
        if proportion<0.5:
            ind2low.append(0)
            proportion=proportion*2
        else:
            ind2low=[i for i in range(1,y.shape[1])]
            proportion=(1-proportion)*2

        new_X=np.array([], ndmin=2)
        new_y=np.array([], ndmin=2)
        for clas in ind2low:
            for ind, num in zip(y.transpose()[clas],range(len(y.transpose()[clas]))):
                if ind>0.5:
                    rnd=random.random()
                    if rnd < proportion:
                        if new_X!=np.array([], ndmin=2):
                            #print(ind, rnd, new_y.shape,new_X.shape[0])
                            tX=np.ndarray(shape=(1,X[num].shape[0]), buffer=X[num].copy())
                            new_X=np.concatenate((new_X,tX), axis=0)
                            ty=np.ndarray(shape=(1,y[num].shape[0]), buffer=y[num].copy(), dtype=int)
                            new_y=np.concatenate((new_y,ty), axis=0)
                        else:
                            new_X=np.ndarray(shape=(1,X[num].shape[0]), buffer=X[num].copy())
                            new_y=np.ndarray(shape=(1,y[num].shape[0]), buffer=y[num].copy(), dtype=int)
                else:
                    if new_X!=np.array([], ndmin=2):
                        tX=np.ndarray(shape=(1,X[num].shape[0]), buffer=X[num].copy())
                        new_X=np.concatenate((new_X,tX), axis=0)
                        ty=np.ndarray(shape=(1,y[num].shape[0]), buffer=y[num].copy(), dtype=int)
                        new_y=np.concatenate((new_y,ty), axis=0)
                    else:
                        new_X=np.ndarray(shape=(1,X[num].shape[0]), buffer=X[num].copy())
                        new_y=np.ndarray(shape=(1,y[num].shape[0]), buffer=y[num].copy(), dtype=int)
        return new_X, new_y

    def make_drift_list(X,y,proportion=0.5):
        #index=[]
        #for key in range(len(scipy.unique(y))):
        #    index.append(key)
        #y=MultiLabelBinarizer(classes=index).fit_transform([[y_p] for y_p in y])
        ind_set=scipy.unique(y)
        if proportion<0.5:
            ind2low=set([0])
            proportion=proportion*2
        else:
            ind2low=set([i for i in range(1,len(ind_set))])
            proportion=(1-proportion)*2

        new_X=np.array([], ndmin=2)
        new_y=[]
        for clas in ind_set:
            for ind, num in zip(y,range(len(y))):
                if ind in ind2low:
                    rnd=random.random()
                    if rnd < proportion:
                        if new_X!=np.array([], ndmin=2):
                            tX=np.ndarray(shape=(1,X[num].shape[0]), buffer=X[num].copy())
                            new_X=np.concatenate((new_X,tX), axis=0)
                            new_y.append(ind)
                        else:
                            new_X=np.ndarray(shape=(1,X[num].shape[0]), buffer=X[num].copy())
                            new_y.append(ind)
                else:
                    if new_X!=np.array([], ndmin=2):
                        tX=np.ndarray(shape=(1,X[num].shape[0]), buffer=X[num].copy())
                        new_X=np.concatenate((new_X,tX), axis=0)
                        new_y.append(ind)
                    else:
                        new_X=np.ndarray(shape=(1,X[num].shape[0]), buffer=X[num].copy())
                        new_y.append(ind)
        return new_X, new_y

    def _kld(self, p, q):
        """Kullback-Leibler divergence D(P || Q) for discrete distributions when Q is used to approximate P
        Parameters
        p, q : array-like, dtype=float, shape=n
        Discrete probability distributions."""
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        return np.sum(np.where(p != 0,p * np.log(p / q), 0))

    def _rae(self, p, q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        return np.sum(np.where(p != 0,np.abs(q-p)/p, 0))

    def _ae(self, p, q):
        #Absolute error
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        return np.average(np.abs(q-p))

    def _emd(self,p,q):
        #Earth Mover’s Distance (Rubner et al., 2000)
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        emd=0
        for i in range(1,len(p)):
            #emd+=np.abs(np.sum(q[0:i])-np.sum(p[0:i]))
            emd+=np.sum(np.abs(q[0:i]-p[0:i]))
        return emd

    def _divergence_bin(self,p,q,func=''):
        if func=='':func=self._kld
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        #print(p,q)
        klds=[]
        for p_i,q_i in zip(p,q):
            klds.append(func([p_i,1-p_i], [q_i,1-q_i]))
        #print(len(_klds))
        #_avg=np.average(_klds)
        return klds#_avg

    def _multi_prevalence(self, y):
        prevalence=[]
        prevalence_smooth=[]
        eps=1/(2*y.shape[0])
        if isinstance(y,csr_matrix):
            for _col in range(y.shape[1]):
                prevalence.append(y.getcol(_col).nnz)
            prevalence=prevalence/(np.sum(prevalence))
            for _val in prevalence: # perform smoothing
                prevalence_smooth.append(_val+eps)
            prevalence_smooth=prevalence_smooth/(np.sum(prevalence)+eps*y.shape[1])
        elif isinstance(y, np.ndarray):
            if len(y.shape)==1:
                yt=MultiLabelBinarizer(classes=self.classes).fit_transform([[y_p] for y_p in y]).transpose()
                for col in yt:
                    prevalence.append(np.sum(col))
                prevalence=prevalence/(np.sum(prevalence))
                for val in prevalence: # perform smoothing
                    prevalence_smooth.append(val+eps)
                prevalence_smooth=prevalence_smooth/(np.sum(prevalence)+eps*yt.shape[0])
            elif len(y.shape)==2:
                yt=y.transpose()
                for col in yt:
                    prevalence.append(np.sum(col))
                prevalence=prevalence/(np.sum(prevalence))
                for val in prevalence: # perform smoothing
                    prevalence_smooth.append(val+eps)
                prevalence_smooth=prevalence_smooth/(np.sum(prevalence)+eps*yt.shape[0])
        return prevalence_smooth

    def _bin_prevalence(self, y):
        prevalence=[]
        if isinstance(y,csr_matrix):
            eps=1/(2*y.shape[0])
            for col in range(y.shape[1]):
                prevalence.append((y.getcol(col).nnz+eps)/(eps*y.shape[1]+y.shape[0]))
            prevalence=np.asarray(prevalence, dtype=np.float)
        elif isinstance(y,list):
            eps=1/(2*len(y))
            yt=MultiLabelBinarizer(classes=self.classes).fit_transform([[y_p] for y_p in y]).transpose()
            for col in range(yt.shape[0]):
                prevalence.append((np.sum(yt[col])+eps)/(eps*yt.shape[0]+yt.shape[1]))
            prevalence=np.asarray(prevalence, dtype=np.float)
        elif isinstance(y, np.ndarray):
            eps=1/(2*y.shape[0])
            if len(y.shape)==1:
                #print(self.classes, 'Variable "y" should have more then 1 dimension. Use MultiLabelBinarizer()')
                yt=MultiLabelBinarizer(classes=self.classes).fit_transform([[y_p] for y_p in y]).transpose()
            elif len(y.shape)==2:
                yt=y.transpose()
            for col in range(yt.shape[0]):
                prevalence.append((np.sum(yt[col])+eps)/(eps*yt.shape[0]+yt.shape[1]))
            prevalence=np.asarray(prevalence, dtype=np.float)
        return prevalence

    def _bin_prevalence_prob(self, y):
        y = np.asarray(y, dtype=np.float).T
        eps=1/(2*y.shape[1])
        prevalence=[]
        print(self.model.intercept_[0][0], self.model.coef_)
        for col in y:
            nnz=0
            for elem in col:
                if elem>=self.model.intercept_:
                    nnz+=1
            prevalence.append((nnz+eps)/(eps*y.shape[0]+y.shape[1]))
        return prevalence

    def __clean_dir(self, dir):
        for name in os.listdir(dir):
            file = os.path.join(dir, name)
            if not os.path.islink(file) and not os.path.isdir(file):
                os.remove(file)

    def __split_by_prevalence(self):
        [csr, y, y_names]=self._read_pickle(self._train_file)
        _prevalence=self.method_prev(y)
        ly=y.shape[1]
        _VLP=[]
        _LP=[]
        _HP=[]
        _VHP=[]
        _col=0
        for _val in _prevalence:
            if _val < 0.01:
                for _i in range(4):
                    _VLP.append(_col+ly*_i)
            elif _val>=0.01 and _val<0.05:
                for _i in range(4):
                    _LP.append(_col+ly*_i)
            elif _val>=0.05 and _val<0.1:
                for _i in range(4):
                    _HP.append(_col+ly*_i)
            elif _val>=0.1:
                for _i in range(4):
                    _VHP.append(_col+ly*_i)
            _col+=1
        return [0, _VLP, _LP, _HP, _VHP]

    def __split_by_distribution_drift(self):#pickle_QuantRCV1
        [csr, y, y_names]=self._read_pickle(self._train_file)
        pr_train=self.method_prev(y)
        _arrange=[]
        j=0
        for _test_file in self._test_files:
            [csr1, y1, y1_names] = self._read_pickle(_test_file)
            pr_test=self.method_prev(y1)
            #_arrange.append((_j,self.kld_bin(pr_test, pr_train)))
            for _i in range(len(pr_train)):
                _arrange.append((j, self._kld([pr_test[_i], 1-pr_test[_i]], [pr_train[_i], 1-pr_train[_i]])))
                j=j+1
        _arrange_sorted=sorted(_arrange, key=operator.itemgetter(1))
        _VLD=[_x[0] for _x in _arrange_sorted[:len(y_names)]]
        _LD=[_x[0] for _x in _arrange_sorted[len(y_names):2*len(y_names)]]
        _HD=[_x[0] for _x in _arrange_sorted[2*len(y_names):3*len(y_names)]]
        _VHD=[_x[0] for _x in _arrange_sorted[3*len(y_names):]]
        return [_arrange, _VLD, _LD, _HD, _VHD]

    def _read_pickle(self, file):
        print('Read file '+file)
        with open(file, 'rb') as f:
            data = pickle.load(f)
            f.close()
        return data

    def _estimate_cl_indexes(self):#pickle_QuantRCV1
        #[_csr, _y, y_names]=self._read_pickle(self._train_file)
        #_prev_train=self.count_prevalence(_y)
        #model=self.arff.fit(_csr,_y)
        _pr_list=[]
        _y1_list=[]
        for _test_file in self._test_files:
            [_csr1, _y1, _y1_names] = self._read_pickle(_test_file)
            _y1_list.append(_y1)
            _pr_list.append(self.model.predict(_csr1))
        with open(self.prefix+'cl_indexes_'+self.dir_name+'.pickle', 'wb') as f:
            print(self.prefix+'cl_indexes_'+self.dir_name+'.pickle')
            pickle.dump([_y, _y1_list, _pr_list, _test_files, y_names], f)
            f.close()
        names_ = [_y, _y1_list, _pr_list, _test_files, y_names]
        return names_

    def __subset(self, _inp_set, _indexes):
        _sub_set=[]
        for _i in _indexes:
            _sub_set.append(_inp_set[_i])
        #_sub_set=_sub_set/np.sum(_sub_set)
        return _sub_set

    def __count_splited_KLD(self, _part, _prev_test, _prev_test_estimate):
        split_by=[np.average(self._divergence_bin(self.__subset(_prev_test,_part[1]), self.__subset(_prev_test_estimate,_part[1]))),
        np.average(self._divergence_bin(self.__subset(_prev_test,_part[2]), self.__subset(_prev_test_estimate,_part[2]))),
        np.average(self._divergence_bin(self.__subset(_prev_test,_part[3]), self.__subset(_prev_test_estimate,_part[3]))),
        np.average(self._divergence_bin(self.__subset(_prev_test,_part[4]), self.__subset(_prev_test_estimate,_part[4]))),
        np.average(self._divergence_bin(_prev_test, _prev_test_estimate))]
        return split_by

    def __count_ttest(self, _prev_test, _prev_test_estimate1, _prev_test_estimate2):
        _kld_1=self._divergence_bin(_prev_test, _prev_test_estimate1)
        _kld_2=self._divergence_bin(_prev_test, _prev_test_estimate2)
        tt=stats.ttest_rel(_kld_1, _kld_2)
        return tt

    def _classify_and_count(self, _y_test):
        #_prev_test=[]
        #for _y_test in y_list:# Test files loop
        #    if is_prob:
        #        _prev_test=np.concatenate((_prev_test,self.method_prev(_y_test)), axis=1)
        #    else:
        #        _prev_test=np.concatenate((_prev_test,self.method_prev(_y_test)), axis=1)
        _prev_test=self.method_prev(_y_test)
        return _prev_test

    def _count_diff1(self, _prev_test, _prev_test_estimate, _num_iter):
        _parts_P=self.__split_by_prevalence()
        _parts_D=self.__split_by_distribution_drift()
        kld_bin=self._divergence_bin(_prev_test, _prev_test_estimate)
        print('\t\t\t VLP \t\t\t LP \t\t\t HP \t\t\t VHP \t\t\t total')
        print(np.average(self.__subset(kld_bin, _parts_P[1])), np.average(self.__subset(kld_bin,_parts_P[2])),\
        np.average(self.__subset(kld_bin,_parts_P[3])), np.average(self.__subset(kld_bin,_parts_P[4])), np.average(kld_bin))
        print('\t\t\t VLD \t\t\t LD \t\t\t HD \t\t\t VHD \t\t\t total')
        print(np.average(self.__subset(kld_bin, _parts_D[1])), np.average(self.__subset(kld_bin,_parts_D[2])),\
        np.average(self.__subset(kld_bin,_parts_D[3])), np.average(self.__subset(kld_bin,_parts_D[4])), np.average(kld_bin))
        print('\t\t\t VLP \t\t\t LP \t\t\t HP \t\t\t VHP \t\t\t total')
        print(np.average(self.__subset(_num_iter, _parts_P[1])), np.average(self.__subset(_num_iter,_parts_P[2])),\
        np.average(self.__subset(_num_iter,_parts_P[3])), np.average(self.__subset(_num_iter,_parts_P[4])), np.average(_num_iter))
        print('\t\t\t VLD \t\t\t LD \t\t\t HD \t\t\t VHD \t\t\t total')
        print(np.average(self.__subset(_num_iter, _parts_D[1])), np.average(self.__subset(_num_iter,_parts_D[2])),\
        np.average(self.__subset(_num_iter,_parts_D[3])), np.average(self.__subset(_num_iter,_parts_D[4])), np.average(_num_iter))
        return 0

    def _count_diff(self, _prev_test, _prev_test_estimate):
        _parts_D=self.__split_by_distribution_drift()
        _parts_P=self.__split_by_prevalence()
        #print(len(_parts_P[1]),len(_parts_P[2]),len(_parts_P[3]),len(_parts_P[4]))
        _kld_P=self.__count_splited_KLD(_parts_P, _prev_test, _prev_test_estimate)
        print('\t\t\t\t VLP \t\t\t\t LP \t\t\t\t HP \t\t\t\t VHP \t\t\t\t total \n', _kld_P)
        _kld_D=self.__count_splited_KLD(_parts_D, _prev_test, _prev_test_estimate)
        print('\t\t\t\t VLD \t\t\t\t LD \t\t\t\t HD \t\t\t\t VHD \t\t\t\t total \n', _kld_D)
        return _kld_P[4]

    def _unite_cl_prob(self):
        #read probabilities from separate files and aggregate it to one file
        [_csr, _y, y_names]=self._read_pickle(self._train_file)
        _train_file, _test_files=self.arff.read_dir(self.prefix+'cl_prob_'+self.dir_name)
        _prob_list=[]
        for _test_file in _test_files:
            with open(_test_file, 'rb') as f:
                _prob = pickle.load(f)
                f.close()
            _prob_list.append(_prob)
        _y1_list=[]
        for _test_file1 in self._test_files:
            [_csr1, _y1, _y1_names] = self._read_pickle(_test_file1)
            _y1_list.append(_y1)
        with open('texts/cl_prob_'+self.dir_name+'.pickle', 'wb') as f:
            pickle.dump([_y, _y1_list, _prob_list, self._test_files, _y1_names], f)
            f.close()
        return [_y, _y1_list, _prob_list, self._test_files, _y1_names]

    def _estimate_cl_prob(self):
        try:
            with open('texts/ml_model_'+self.dir_name+'.pickle', 'rb') as f:
                self.model = pickle.load(f)
                f.close()
        except:
            [_csr, _y, y_names]=self._read_pickle(self._train_file)
            _prev_train=self.count_prevalence(_y)
            model=self.model#self.arff.fit(_csr,_y)
            with open('texts/ml_model_'+self.dir_name+'.pickle', 'wb') as f:
                pickle.dump(model, f)
                f.close()

        _prob_list=[]
        _y1_list=[]
        for _t in range(len(self._test_files)):# range(42,52):
            _test_file=self._test_files[_t]
            [_csr1, _y1, _y1_names] = self._read_pickle(_test_file)
            _y1_list.append(_y1)
            _prob=model.predict_proba(_csr1)
            _prob_list.append(_prob)
            with open('texts/cl_prob_'+_test_file.rstrip('.arff.pickle').lstrip('texts/pickle_')+'.cl_prob', 'wb') as f:
                pickle.dump(_prob, f)
                f.close()
        with open('texts/cl_prob_'+self.dir_name+'.pickle', 'wb') as f:
            pickle.dump([_y, _y1_list, _prob_list, self._test_files, _y1_names], f)
            f.close()
        return [_y, _y1_list, _prob_list, self._test_files, y_names]

    def _prob_classify_and_count(self, pred_prob):
        #avr_prob=[]
        #for pred_prob in pred_prob_list:
        #    avr_prob=np.concatenate((avr_prob,np.average(pred_prob, axis=0)))
        #print('PCC',avr_prob)
        return np.average(pred_prob, axis=0)

    def _exp_max(self, y_train, pred_prob, stop_delta=0.1):
        pr_train=self._bin_prevalence(y_train)
        pr_all=[]
        pr_s=pr_train.copy()
        prob_t=pred_prob.T
        prob_t_s =prob_t.copy()
        delta=1
        delta_s=1
        count=0
        while delta>stop_delta and delta<=delta_s and count<100:
            for cl_n in range(len(pr_train)):#Category
                prob_t_s[cl_n]=prob_t[cl_n].copy()*(pr_s[cl_n]/pr_train[cl_n])  #E step
            prob_t_s=normalize(prob_t_s, norm='l1',axis=0)                      #E step
            pr_s1=np.average(prob_t_s, axis=1)                                  #M step
            #pr_s1=self._adj_classify_and_count([prob_t_s.transpose()],is_prob=True)
            delta_s=delta
            #delta=np.max(np.abs(pr_s1-pr_s))
            delta=self._ae(pr_s,pr_s1)
            #print('pr_s1',pr_s1, delta)
            #print(prob_t_s)
            #pr_train=pr_s.copy()
            #prob_t=prob_t_s.copy()
            pr_s=pr_s1.copy()
            count=count+1
        if np.max(pr_s)>0.99: pr_s=np.average(prob_t, axis=1)
        return pr_s

    def _expectation_maximization(self, y_train, pred_prob, stop_delta=0.1):#_indexes
        #[y_train, y_test_list, pred_prob_list, test_files, y_names]=_indexes
        #print(pred_prob_list[0][1])
        pr_train=self._bin_prevalence(y_train)
        pr_all=[]
        num_iter=[]
        test_num=0#0..3 len(_y_test_list)
        pr_c=pr_train.copy()

        prob=pred_prob.T
        for cl_n in range(len(pr_train)):#Category
            #print('Test set N %s, class number %s' %(test_num, cl_n))
            iter=0
            _delta=1
            while _delta>stop_delta:
                pr_c_x=[]
                _j=0
                for pr_c_xk in prob[cl_n]:#xk in category c
            #Step E
                    pr_c_x_k=(pr_c[cl_n]/pr_train[cl_n]*pr_c_xk)/(((1-pr_c[cl_n])/(1-pr_train[cl_n]))*(1-pr_c_xk)+pr_c[cl_n]/pr_train[cl_n]*pr_c_xk)
                    pr_c_x.append(pr_c_x_k)
                    _j+=1
            #Step M
                pr_c_new=np.average(pr_c_x)#np.average(_prob[cl_n])
                _delta=np.abs(pr_c_new-pr_c[cl_n])
                #print('_delta',_delta)
                #pr_train[cl_n]=pr_c[cl_n]
                #prob[cl_n]=pr_c_x_k
                pr_c[cl_n]=pr_c_new
                iter+= 1
            num_iter.append(iter)
            if np.max([pr_c[cl_n],1-pr_c[cl_n]])>0.99: pr_c[cl_n]=np.average(prob[cl_n])
        return pr_c #,num_iter

    def _cost_sens_learning(self, X_test, stop_delta=0.00001, class_weight_start='auto'):
        pred_prev_train=self._classify_and_count(self.y_train)
        pred_prev0=pred_prev_train.copy()
        model=self.__classificator(class_weight=class_weight_start)#class_weight={0:1,1:1})##
        model.fit(self.X_train, self.y_train)
        pred_prev1=np.average(model.predict_proba(X_test), axis=0)#
        #pred_prev1=self._classify_and_count(model.predict(X_test))
        delta1=0
        delta2=0
        d_delta1=0
        d_delta2=0
        for i in range(10):
            #print('pred_prev0',pred_prev0)
            #print('pred_prev1',pred_prev1)
            #print(pred_prev1/pred_prev_train)
            #print(delta2)
            class_weight=dict(zip(self.classes, pred_prev1/pred_prev_train))

            model=self.__classificator(class_weight=class_weight)
            model.fit(self.X_train, self.y_train)
            pred_prev2=np.average(model.predict_proba(X_test), axis=0)#
            #pred_prev2=self._classify_and_count(model.predict(X_test))#
            delta1=delta2
            delta2=self._ae(pred_prev1,pred_prev2)
            d_delta3=abs(delta2-delta1)
            if delta2<stop_delta or d_delta3>d_delta2 and d_delta2>d_delta1 and d_delta1!=0:
                #print('dd',d_delta1, d_delta2,d_delta3)
                break
            d_delta1=d_delta2
            d_delta2=d_delta3
            #print(pred_prev2[0],'\t', delta1)
            #if delta2<stop_delta:
            #    break
            pred_prev0=pred_prev1.copy()
            pred_prev1=pred_prev2.copy()
        #print('pred_prev1',pred_prev1)
        return pred_prev1

    def cost_sens_learning_out(self, X_test, stop_delta=0.00001, class_weight_start='auto'):
        pred_prev_train=self._classify_and_count(self.y_train)
        pred_prev0=pred_prev_train.copy()
        model=self.__classificator(class_weight=class_weight_start)#class_weight={0:1,1:1})##
        model.fit(self.X_train, self.y_train)
        pred_prev1=np.average(model.predict_proba(X_test), axis=0)#
        #pred_prev1=self._classify_and_count(model.predict(X_test))
        delta1=0
        delta2=0
        d_delta1=0
        d_delta2=0
        for i in range(10):
            #print('pred_prev0',pred_prev0)
            #print('pred_prev1',pred_prev1)
            #print(pred_prev1/pred_prev_train)
            #print(delta2)
            class_weight=dict(zip(self.classes, pred_prev1/pred_prev_train))

            model=self.__classificator(class_weight=class_weight)
            model.fit(self.X_train, self.y_train)
            pred_prev2=np.average(model.predict_proba(X_test), axis=0)#
            #pred_prev2=self._classify_and_count(model.predict(X_test))#
            delta1=delta2
            delta2=self._ae(pred_prev1,pred_prev2)
            d_delta3=abs(delta2-delta1)
            if delta2<stop_delta or d_delta3>d_delta2 and d_delta2>d_delta1 and d_delta1!=0:
                #print('dd',d_delta1, d_delta2,d_delta3)
                break
            d_delta1=d_delta2
            d_delta2=d_delta3
            #print(pred_prev2[0],'\t', delta1)
            #if delta2<stop_delta:
            #    break
            pred_prev0=pred_prev1.copy()
            pred_prev1=pred_prev2.copy()

        #print('pred_prev1',pred_prev1)
        return model.predict(X_test),pred_prev1

    def __conditional_probability(self,p1,p2,val1,val2):
        c=0
        for _i in range(len(p1)):
            if p1[_i]==val1 and p2[_i]==val2:
                c=c+1
        return c/len(p1)

    def __kfold_tp_fp(self, X, y, n_folds=2):
        #return true positive rate and false positive rate arrays

        #if isinstance(X, csr_matrix) and isinstance(y, csr_matrix):
        #    X=X.toarray()
        #    y=y.toarray()
        #elif isinstance(X, csr_matrix) and isinstance(y, np.ndarray):
        #    X=X.toarray()
        #    y=MultiLabelBinarizer(classes=self.classes).fit_transform([[y_p] for y_p in y])
        #elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        #    if len(y.shape)==1:
        #        y=MultiLabelBinarizer(classes=self.classes).fit_transform([[y_p] for y_p in y])
        #    elif len(y.shape)==2:
        #        pass
        if isinstance(y, list):
            y=np.asarray(y)

        try:
            with open(self.prefix+self.dir_name+'/'+str(n_folds)+'FCV.pickle', 'rb') as f:
                [tp_av, fp_av] = pickle.load(f)
        except:
            _kf=KFold(y.shape[0],n_folds=n_folds)
            tp=[]
            fp=[]
            for train_index, test_index in _kf:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model=self.model
                model=model.fit(X_train, y_train)#arff.fit(X_train, y_train)
                y_predict=model.predict(X_test)
                tp_k=[]
                fp_k=[]
                if len(y.shape)==1:
                    y_test=MultiLabelBinarizer(classes=self.classes).fit_transform([[y_p] for y_p in y_test])
                    y_predict=MultiLabelBinarizer(classes=self.classes).fit_transform([[y_p] for y_p in y_predict])
                elif len(y.shape)==2:
                    pass
                for s_true,s_pred in zip(y_test.T,y_predict.T):
                    tp_k.append(self.__conditional_probability(s_pred, s_true, 1., 1.))#cm[0,0]/len(s_true))
                    fp_k.append(self.__conditional_probability(s_pred, s_true, 1., 0.))#cm[1,0]/len(s_true))#len(s_true))
                tp.append(tp_k)
                fp.append(fp_k)
            tp_av=np.asarray([np.average(tp_k) for tp_k in np.asarray(tp).T])
            fp_av=np.asarray([np.average(fp_k) for fp_k in np.asarray(fp).T])
            with open(self.prefix+self.dir_name+'/'+str(n_folds)+'FCV.pickle', 'wb') as f:
                pickle.dump([tp_av, fp_av], f)
                f.close()
            #print('[tp_av, fp_av] by index',tp_av, fp_av)
        return [tp_av, fp_av]

    def __kfold_prob_tp_fp(self, X, y, n_folds=2):
        # if isinstance(X, csr_matrix) and isinstance(y, np.ndarray):
        #     X=X.toarray()
        # elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        #     if len(y.shape)==1:
        #         y=MultiLabelBinarizer(classes=self.classes).fit_transform([[y_p] for y_p in y])
        #     elif len(y.shape)==2:
        #         pass
        if isinstance(y, list):
            y=np.asarray(y)

        try:
            with open(self.prefix+self.dir_name+'/'+str(n_folds)+'FCV_prob.pickle', 'rb') as f:
                [tp_av, fp_av] = pickle.load(f)
        except:
            kf=KFold(y.shape[0],n_folds=n_folds)
            TP_avr=[]
            FP_avr=[]
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model=self.model
                model=model.fit(X_train, y_train)
                y_predict=model.predict(X_test)
                y_prob_predict=model.predict_proba(X_test)
                TP=[]
                FP=[]
                if len(y.shape)==1:
                    y_predict=MultiLabelBinarizer(classes=self.classes).fit_transform([[y_p] for y_p in y_predict])
                elif len(y.shape)==2:
                    pass
                for class_ind, class_prob in zip(y_predict.transpose(), y_prob_predict.transpose()):
                    TP_class=[]
                    FP_class=[]
                    for ind, prob in zip(class_ind, class_prob):
                        if ind==1: TP_class.append(prob)
                        elif ind==0: FP_class.append(prob)
                    TP.append(np.sum(TP_class)/len(class_ind))
                    FP.append(np.sum(FP_class)/len(class_ind))
                TP_avr.append(TP)
                FP_avr.append(FP)
            tp_av, fp_av=np.average(TP_avr, axis=0), np.average(FP_avr, axis=0)
            with open(self.prefix+self.dir_name+'/'+str(n_folds)+'FCV_prob.pickle', 'wb') as f:
                pickle.dump([tp_av, fp_av], f)
                f.close()
            #print('tp, fp by prob', tp_av, fp_av)
        return [tp_av, fp_av]

    def _adj_classify_and_count(self, y_pred, is_prob=False):
        [tp_av, fp_av]=self.kfold_results

        if is_prob:
            pr=np.average(y_pred,axis=0)
        else:
            pr=self.method_prev(y_pred)
        try:
            pred=(pr-fp_av)/(tp_av-fp_av)
            if np.min(pred)>=0:
                pred=normalize(pred, norm='l1', axis=1)[0]
            else:
                #print(pred)
                #print(pr,tp_av,fp_av)
                pred=pr
        except:
            print(pr,tp_av,fp_av)
            pred=pr
        return pred

    def _process_pipeline(self):
        #Warning! Processing can takes a long period. We recommend to perform it step by step
        #pa=Parse_ARFF()
        #pa.convert_arff(QuantOHSUMED, is_predict=False)
        #q=Quantification('QuantOHSUMED')
        #q.process_pipeline()
        #####################################################
        [self.X_train, self.y_train, y_names]=self._read_pickle(self._train_file)
        self.fit(self.X_train, self.y_train)
        #[y_train, y_test_list, y_pred_list, test_files, y_names]=self._estimate_cl_indexes()
        [y_train,y_test_list,y_pred_list,test_files, y_names]=self._read_pickle('texts/cl_indexes_'+self.dir_name+'.pickle')
        td=self._classify_and_count(y_test_list)
        ed1=self._classify_and_count(y_pred_list)

        ed2=self._adj_classify_and_count(self.X_train, self.y_train, y_pred_list)

        self._estimate_cl_prob()
        self._unite_cl_prob()
        [y_train,y_test_list,pred_prob_list,test_files, y_names]=self._read_pickle('texts/cl_prob_'+self.dir_name+'.pickle')
        ed3=self._classify_and_count(pred_prob_list, is_prob=True)
        ed4=self._prob_classify_and_count(pred_prob_list)
        ed5, num_iter=self._expectation_maximization(self.y_train,pred_prob_list, 0.1)
        self._count_diff(td,ed4)
        self._count_diff1(td,ed5, num_iter)
