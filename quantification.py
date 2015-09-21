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
        self.method_prev=self.bin_prevalence#.bin_prevalence #.multi_prevalence

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
        _klds=[]
        for _i in range(len(p)):
            _klds.append(self.kld([p[_i],1-p[_i]], [q[_i],1-q[_i]]))
        #print(len(_klds))
        _avg=np.average(_klds)
        return _avg

    def multi_prevalence(self, _y):
        _prevalence=[]
        _prevalence_smooth=[]
        _eps=1/(2*_y.shape[0])
        for _col in range(_y.shape[1]):
            _prevalence.append(_y.getcol(_col).nnz)
        _prevalence=_prevalence/(np.sum(_prevalence))

        for _val in _prevalence: # perform smoothing
            _prevalence_smooth.append(_val+_eps)
        _prevalence_smooth=_prevalence_smooth/(np.sum(_prevalence)+_eps*_y.shape[1])
        return _prevalence_smooth

    def bin_prevalence(self, _y):
        _eps=1/(2*_y.shape[0])
        _prevalence=[]
        for _col in range(_y.shape[1]):
            _prevalence.append((_y.getcol(_col).nnz+_eps)/(_eps*_y.shape[1]+_y.shape[0]))
        _prevalence=np.asarray(_prevalence, dtype=np.float)
        return _prevalence

    def split_by_prevalence(self):
        _train_file, _test_files=self.arff.read_dir(self.dir_name)
        [_csr, _y, y_names]=self.read_pickle(_train_file)
        _prevalence=self.method_prev(_y)
        _VLP=[]
        _LP=[]
        _HP=[]
        _VHP=[]
        _col=0
        for _val in _prevalence:
            if _val < 0.01:
                for _i in range(1,5):
                    _VLP.append(_col*_i)
            elif _val>=0.01 and _val<0.05:
                for _i in range(1,5):
                    _LP.append(_col*_i)
            elif _val>=0.05 and _val<0.1:
                for _i in range(1,5):
                    _HP.append(_col*_i)
            elif _val>=0.1:
                for _i in range(1,5):
                    _VHP.append(_col*_i)
            _col+=1
        return [0, _VLP, _LP, _HP, _VHP]

    def split_by_distribution_drift(self):#pickle_QuantRCV1
        _train_file, _test_files=self.arff.read_dir(self.dir_name)
        [_csr, _y, y_names]=self.read_pickle(_train_file)
        pr_train=self.method_prev(_y)
        _arrange=[]
        _j=0
        for _test_file in _test_files:
            [_csr1, _y1, _y1_names] = self.read_pickle(_test_file)
            pr_test=self.method_prev(_y1)
            #_arrange.append((_j,self.kld_bin(pr_test, pr_train)))
            for _i in range(len(pr_train)):
                _arrange.append((_j, self.kld([pr_test[_i], 1-pr_test[_i]], [pr_train[_i], 1-pr_train[_i]])))
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

    def estimate_cl_indexes(self):#pickle_QuantRCV1
        _train_file, _test_files=self.arff.read_dir(self.dir_name)
        [_csr, _y, y_names]=self.read_pickle(_train_file)
        #_prev_train=self.count_prevalence(_y)
        model=self.arff.fit(_csr,_y)
        _pr_list=[]
        _y1_list=[]
        for _test_file in _test_files:
            [_csr1, _y1, _y1_names] = self.read_pickle(_test_file)
            _y1_list.append(_y1)
            _pr_list.append(model.predict(_csr1))
        with open('texts/'+self.dir_name+'_cl_indexes', 'wb') as f:
            pickle.dump([_y, _y1_list, _pr_list, _test_files, y_names], f)
        names_ = [_y, _y1_list, _pr_list, _test_files, y_names]
        return names_

    def subset(self, _inp_set, _indexes):
        _sub_set=[]
        for _i in _indexes:
            _sub_set.append(_inp_set[_i])
        _sub_set=_sub_set/np.sum(_sub_set)
        return _sub_set

    def count_splited_KLD(self, _part, _prev_test, _prev_test_estimate):
        split_by_prevalence=[self.kld_bin(self.subset(_prev_test,_part[1]), self.subset(_prev_test_estimate,_part[1])),
        self.kld_bin(self.subset(_prev_test,_part[2]), self.subset(_prev_test_estimate,_part[2])),
        self.kld_bin(self.subset(_prev_test,_part[3]), self.subset(_prev_test_estimate,_part[3])),
        self.kld_bin(self.subset(_prev_test,_part[4]), self.subset(_prev_test_estimate,_part[4])),
        self.kld_bin(_prev_test, _prev_test_estimate)]
        return split_by_prevalence

    def classify_and_count(self, _indexes):
        _prev_test=[]
        _prev_test_estimate=[]
        for _i in range(len(_indexes[1])):
            #print(_i, metrics.classification_report(_indexes[1][_i],_indexes[2][_i]))
            _prev_test=np.concatenate((_prev_test,self.method_prev(_indexes[1][_i])), axis=1)
            _prev_test_estimate=np.concatenate((_prev_test_estimate,self.method_prev(_indexes[2][_i])), axis=1)
        return _prev_test, _prev_test_estimate

    def count_diff(self, _prev_test, _prev_test_estimate):
        _parts_P=self.split_by_distribution_drift()
        _parts_D=self.split_by_prevalence()
        _kld_P=self.count_splited_KLD(_parts_P, _prev_test, _prev_test_estimate)
        print('\t\t\t\t VLP \t\t\t\t LP \t\t\t\t HP \t\t\t\t VHP \t\t\t\t total \n', _kld_P)
        _kld_D=self.count_splited_KLD(_parts_D, _prev_test, _prev_test_estimate)
        print('\t\t\t\t VLD \t\t\t\t LD \t\t\t\t HD \t\t\t\t VHD \t\t\t\t total \n', _kld_D)
        return 0

    def estimate_cl_prob(self):
        _train_file, _test_files=self.arff.read_dir(self.dir_name)
        [_csr, _y, y_names]=self.read_pickle(_train_file)
        #_prev_train=self.count_prevalence(_y)
        model=self.arff.fit(_csr,_y)
        _pr_list=[]
        _y1_list=[]
        for _test_file in _test_files:
            [_csr1, _y1, _y1_names] = self.read_pickle(_test_file)
            _y1_list.append(_y1)
            _pr=model.predict_proba(_csr1)
            _pr_list.append(_pr)
        with open('texts/'+self.dir_name+'_cl_prob', 'wb') as f:
            pickle.dump([_y, _y1_list, _pr_list, _test_files, y_names], f)
        return [_y, _y1_list, _pr_list, _test_files, y_names]

    def expectation_maximization(self, _indexes):
        [_y_train, _y_test_list, _pred_prob_list, _test_files, y_names]=_indexes
        pr_train=self.bin_prevalence(_y_train)
        pr_all=[]
        test_prevalence=[]

        _test_num=0#0..3 len(_y_test_list)
        for _pred_prob in _pred_prob_list:
            print('Test file N', _test_num)
            test_prevalence=np.concatenate((test_prevalence,self.bin_prevalence(_y_test_list[_test_num])), axis=1)
            pr_c=[x for x in pr_train]

            _prob=_pred_prob.T
            for _class_num in range(len(pr_train)):
                print('Test file N %s, class number %s' %(_test_num, _class_num))
                _delta=1
                while _delta>0.0001:#0.00000000000000001:
                    pr_c_x=[]
                    _j=0
                    for pr_c_xk in _prob[_class_num]:
                #Step E
                        pr_c_x_k=(pr_c[_class_num]/pr_train[_class_num]*pr_c_xk)/(((1-pr_c[_class_num])/(1-pr_train[_class_num]))*(1-pr_c_xk)+pr_c[_class_num]/pr_train[_class_num]*pr_c_xk)
                        #if _j==0: print(pr_c[_class_num],pr_train[_class_num], pr_c_xk, pr_c_x_k)
                        pr_c_x.append(pr_c_x_k)
                        _j+=1
                #Step M
                    pr_c_new=np.average(pr_c_x)

                    _delta=np.abs(pr_c_new-pr_c[_class_num])
                    if _class_num==87 and _test_num==3: print(pr_c_new, pr_c[_class_num], _delta, _delta>0.0001)
                    pr_c[_class_num]=pr_c_new
            pr_all=np.concatenate((pr_all,pr_c), axis=1)
            _test_num+=1
        for _j in range(len(test_prevalence)):
            print(test_prevalence[_j], pr_all[_j], pr_train[(_j % 88)],  _j)
        return test_prevalence, pr_all

name='pickle_QuantOHSUMED' #pickle_QuantRCV1 #pickle_QuantOHSUMED
q=Quantification(name)
#indexes=q.estimate_cl_indexes()
#with open('texts/'+name+'_cl_indexes', 'rb') as f:
#    indexes = pickle.load(f)
#td,ed=q.classify_and_count(indexes)
#q.classify_and_count(indexes, q.multi_prevalence)

#e=q.estimate_cl_prob()
with open('texts/'+name+'_cl_prob', 'rb') as f:
    prob = pickle.load(f)
td,ed=q.expectation_maximization(prob)
q.count_diff(td,ed)