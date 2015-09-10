# coding: utf-8
__author__ = 'Nikolay Karpov'

from parse_arff import parse_arff
import pickle
import numpy as np
import operator
from sklearn.preprocessing import normalize
from sklearn import metrics

class Quantification:
    def __init__(self, _dir_name):
        self.arff=parse_arff()
        self.dir_name=_dir_name

    def kld(self, p, q):
        """Kullback-Leibler divergence D(P || Q) for discrete distributions when Q is used to approximate P
        Parameters
        p, q : array-like, dtype=float, shape=n
        Discrete probability distributions."""
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        return np.sum(np.where(p != 0,p * np.log(p / q), 0))

    def kld_bin(self,p,q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        _avg=0
        for _i in range(len(p)):
            _avg+=self.kld([p[_i],1-p[_i]], [q[_i],1-q[_i]])
        _avg=_avg/len(p)
        return _avg

    def count_prevalence(self, _y):
        _VLP=[]
        _LP=[]
        _HP=[]
        _VHP=[]
        _prevalence=[]
        _prevalence_smooth=[]
        _eps=1/(2*_y.shape[0])
        for _col in range(_y.shape[1]):
            _prevalence.append(_y.getcol(_col).nnz)
        _prevalence=_prevalence/(np.sum(_prevalence))

        for _val in _prevalence: # perform smoothing
            _prevalence_smooth.append(_val+_eps)
        _prevalence_smooth=_prevalence_smooth/(np.sum(_prevalence)+_eps*_y.shape[1])
        _col=0
        for _val in _prevalence_smooth:
            if _val < 0.01:
                _VLP.append(_col)
            elif _val>=0.01 and _val<0.05:
                _LP.append(_col)
            elif _val>=0.05 and _val<0.1:
                _HP.append(_col)
            elif _val>=0.1:
                _VHP.append(_col)
            _col+=1
        return [_prevalence_smooth, _VLP, _LP, _HP, _VHP]

    def count_bin_pr(self, _y):
        _VLP=[]
        _LP=[]
        _HP=[]
        _VHP=[]
        _eps=1/(2*_y.shape[0])
        _prevalence=[]
        for _col in range(_y.shape[1]):
            _prevalence.append((_y.getcol(_col).nnz+_eps)/(_eps*_y.shape[1]+_y.shape[0]))
        _col=0
        for _val in _prevalence:
            if _val < 0.01:
                _VLP.append(_col)
            elif _val>=0.01 and _val<0.05:
                _LP.append(_col)
            elif _val>=0.05 and _val<0.1:
                _HP.append(_col)
            elif _val>=0.1:
                _VHP.append(_col)
            _col+=1
        return [_prevalence, _VLP, _LP, _HP, _VHP]

    def split_by_distribution_drift(self):#pickle_QuantRCV1
        _train_file, _test_files=self.arff.read_dir(self.dir_name)
        [_csr, _y, y_names]=self.read_pickle(_train_file)
        pr_train=self.count_bin_pr(_y)
        _arrange=[]
        _j=0
        for _test_file in _test_files:
            [_csr1, _y1, _y1_names] = self.read_pickle(_test_file)
            pr_test=self.count_bin_pr(_y1)
            for _i in range(len(pr_train[0])):
                _arrange.append((_j, self.kld([pr_test[0][_i], 1-pr_test[0][_i]], [pr_train[0][_i], 1-pr_train[0][_i]])))
                _j=_j+1
        _arrange_sorted=sorted(_arrange, key=operator.itemgetter(1))
        _VLD=[_x[0] for _x in _arrange_sorted[:len(y_names)]]
        _LD=[_x[0] for _x in _arrange_sorted[len(y_names):2*len(y_names)]]
        _HD=[_x[0] for _x in _arrange_sorted[2*len(y_names):3*len(y_names)]]
        _VHD=[_x[0] for _x in _arrange_sorted[3*len(y_names):]]
        return [_arrange, _VLD, _LD, _HD, _VHD]

    def read_pickle(self, _file):
        print('Read file '+_file)
        with open(_file, 'rb') as f:
            [_csr, _y, y_names] = pickle.load(f)
        return [_csr, _y, y_names]
    def estimate_cl_indexes(self, _path_name):#pickle_QuantRCV1
        _train_file, _test_files=self.arff.read_dir(_path_name)
        [_csr, _y, y_names]=self.read_pickle(_train_file)
        #_prev_train=self.count_prevalence(_y)
        model=self.arff.fit(_csr,_y)
        _pr_list=[]
        _y1_list=[]
        for _test_file in _test_files:
            [_csr1, _y1, _y1_names] = self.read_pickle(_test_file)
            _y1_list.append(_y1)
            _pr_list.append(model.predict(_csr1))
        with open('texts/'+_path_name+'_cl_indexes', 'wb') as f:
            pickle.dump([_y, _y1_list, _pr_list, _test_files, y_names], f)
        return [_y, _y1_list, _pr_list, _test_files, y_names]

    def subset(self, _inp_set, _indexes):
        _sub_set=[]
        for _i in _indexes:
            _sub_set.append(_inp_set[_i])
        _sub_set=_sub_set/np.sum(_sub_set)
        return _sub_set

    def count_splited_KLD(self, _prev_train, _prev_test, _prev_test_estimate):
        split_by_prevalence=[self.kld_bin(self.subset(_prev_test,_prev_train[1]), self.subset(_prev_test_estimate,_prev_train[1])),
        self.kld_bin(self.subset(_prev_test,_prev_train[2]), self.subset(_prev_test_estimate,_prev_train[2])),
        self.kld_bin(self.subset(_prev_test,_prev_train[3]), self.subset(_prev_test_estimate,_prev_train[3])),
        self.kld_bin(self.subset(_prev_test,_prev_train[4]), self.subset(_prev_test_estimate,_prev_train[4])),
        self.kld_bin(_prev_test, _prev_test_estimate)]
        return split_by_prevalence

    def classify_and_count(self, _indexes):
        _parts=self.split_by_distribution_drift()
        _prev_test=[]
        _prev_test_estimate=[]
        for _i in range(len(_indexes[1])):
            #print(_i, metrics.classification_report(_indexes[1][_i],_indexes[2][_i]))
            _prev_test=_prev_test+self.count_bin_pr(_indexes[1][_i])[0]
            _prev_test_estimate=_prev_test_estimate+self.count_bin_pr(_indexes[2][_i])[0]

        _kld_D=self.count_splited_KLD(_parts, _prev_test, _prev_test_estimate)
        print('VLD - VHD', _kld_D)
        _kld_P=self.count_splited_KLD(self.count_bin_pr(_indexes[0]), _prev_test, _prev_test_estimate)
        print('VLP - VHP', _kld_P)

q=Quantification('pickle_QuantOHSUMED')
#e=q.estimate_cl_indexes('pickle_QuantOHSUMED')
with open('texts/pickle_QuantOHSUMED_cl_indexes', 'rb') as f:
    _indexes = pickle.load(f)
q.classify_and_count(_indexes)