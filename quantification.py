# coding: utf-8
__author__ = 'Nikolay Karpov'

from parse_arff import Parse_ARFF
import pickle
import numpy as np
import operator
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn import metrics
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC, SVC
from scipy import stats

class Quantification:
    def __init__(self, _dir_name):
        self.prefix='texts/'
        self.arff=Parse_ARFF()
        self.dir_name=_dir_name
        self.method_prev=self.bin_prevalence#.bin_prevalence or .multi_prevalence
        self._train_file, self._test_files=self.arff.read_dir(self.prefix+'pickle_'+_dir_name)

    def kld(self, p, q):
        """Kullback-Leibler divergence D(P || Q) for discrete distributions when Q is used to approximate P
        Parameters
        p, q : array-like, dtype=float, shape=n
        Discrete probability distributions."""
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        return np.sum(np.where(p != 0,p * np.log(p / q), 0))

    def rae(self, p, q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        return np.where(p != 0,np.abs(q-p)/p, 0)

    def kld_bin(self,p,q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        _klds=[]
        for _i in range(len(p)):
            _klds.append(self.kld([p[_i],1-p[_i]], [q[_i],1-q[_i]]))
        #print(len(_klds))
        #_avg=np.average(_klds)
        return _klds#_avg

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

    def bin_prevalence_prob(self, _y):
        _y = np.asarray(_y, dtype=np.float).T
        _eps=1/(2*_y.shape[1])
        _prevalence=[]
        for _col in _y:
            nnz=0
            for _elem in _col:
                if _elem>=0.5:
                    nnz+=1
            _prevalence.append((nnz+_eps)/(_eps*_y.shape[0]+_y.shape[1]))
        return _prevalence

    def split_by_prevalence(self):
        [_csr, _y, y_names]=self._read_pickle(self._train_file)
        _prevalence=self.method_prev(_y)
        ly=_y.shape[1]
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

    def split_by_distribution_drift(self):#pickle_QuantRCV1
        [_csr, _y, y_names]=self._read_pickle(self._train_file)
        pr_train=self.method_prev(_y)
        _arrange=[]
        _j=0
        for _test_file in self._test_files:
            [_csr1, _y1, _y1_names] = self._read_pickle(_test_file)
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

    def _read_pickle(self, _file):
        print('Read file '+_file)
        with open(_file, 'rb') as f:
            data = pickle.load(f)
            f.close()
        return data

    def estimate_cl_indexes(self):#pickle_QuantRCV1
        [_csr, _y, y_names]=self._read_pickle(self._train_file)
        #_prev_train=self.count_prevalence(_y)
        model=self.arff.fit(_csr,_y)
        _pr_list=[]
        _y1_list=[]
        for _test_file in self._test_files:
            [_csr1, _y1, _y1_names] = self._read_pickle(_test_file)
            _y1_list.append(_y1)
            _pr_list.append(model.predict(_csr1))
        with open(self.prefix+'cl_indexes_'+self.dir_name+'.pickle', 'wb') as f:
            print(self.prefix+'cl_indexes_'+self.dir_name+'.pickle')
            pickle.dump([_y, _y1_list, _pr_list, _test_files, y_names], f)
            f.close()
        names_ = [_y, _y1_list, _pr_list, _test_files, y_names]
        return names_

    def subset(self, _inp_set, _indexes):
        _sub_set=[]
        for _i in _indexes:
            _sub_set.append(_inp_set[_i])
        #_sub_set=_sub_set/np.sum(_sub_set)
        return _sub_set

    def count_splited_KLD(self, _part, _prev_test, _prev_test_estimate):
        split_by=[np.average(self.kld_bin(self.subset(_prev_test,_part[1]), self.subset(_prev_test_estimate,_part[1]))),
        np.average(self.kld_bin(self.subset(_prev_test,_part[2]), self.subset(_prev_test_estimate,_part[2]))),
        np.average(self.kld_bin(self.subset(_prev_test,_part[3]), self.subset(_prev_test_estimate,_part[3]))),
        np.average(self.kld_bin(self.subset(_prev_test,_part[4]), self.subset(_prev_test_estimate,_part[4]))),
        np.average(self.kld_bin(_prev_test, _prev_test_estimate))]
        return split_by

    def count_ttest(self, _prev_test, _prev_test_estimate1, _prev_test_estimate2):
        _kld_1=self.kld_bin(_prev_test, _prev_test_estimate1)
        _kld_2=self.kld_bin(_prev_test, _prev_test_estimate2)
        tt=stats.ttest_rel(_kld_1, _kld_2)
        return tt

    def classify_and_count(self, _y_list, is_prob=False):
        _prev_test=[]
        for _y_test in _y_list:# Test files loop
            if is_prob:
                _prev_test=np.concatenate((_prev_test,self.bin_prevalence_prob(_y_test)), axis=1)
            else:
                _prev_test=np.concatenate((_prev_test,self.bin_prevalence(_y_test)), axis=1)
        return _prev_test

    def count_diff1(self, _prev_test, _prev_test_estimate, _num_iter):
        _parts_P=self.split_by_prevalence()
        _parts_D=self.split_by_distribution_drift()
        kld_bin=self.kld_bin(_prev_test, _prev_test_estimate)
        print('\t\t\t VLP \t\t\t LP \t\t\t HP \t\t\t VHP \t\t\t total')
        print(np.average(self.subset(kld_bin, _parts_P[1])), np.average(self.subset(kld_bin,_parts_P[2])),\
        np.average(self.subset(kld_bin,_parts_P[3])), np.average(self.subset(kld_bin,_parts_P[4])), np.average(kld_bin))
        print('\t\t\t VLD \t\t\t LD \t\t\t HD \t\t\t VHD \t\t\t total')
        print(np.average(self.subset(kld_bin, _parts_D[1])), np.average(self.subset(kld_bin,_parts_D[2])),\
        np.average(self.subset(kld_bin,_parts_D[3])), np.average(self.subset(kld_bin,_parts_D[4])), np.average(kld_bin))
        print('\t\t\t VLP \t\t\t LP \t\t\t HP \t\t\t VHP \t\t\t total')
        print(np.average(self.subset(_num_iter, _parts_P[1])), np.average(self.subset(_num_iter,_parts_P[2])),\
        np.average(self.subset(_num_iter,_parts_P[3])), np.average(self.subset(_num_iter,_parts_P[4])), np.average(_num_iter))
        print('\t\t\t VLD \t\t\t LD \t\t\t HD \t\t\t VHD \t\t\t total')
        print(np.average(self.subset(_num_iter, _parts_D[1])), np.average(self.subset(_num_iter,_parts_D[2])),\
        np.average(self.subset(_num_iter,_parts_D[3])), np.average(self.subset(_num_iter,_parts_D[4])), np.average(_num_iter))
        return 0

    def count_diff(self, _prev_test, _prev_test_estimate):
        _parts_D=self.split_by_distribution_drift()
        _parts_P=self.split_by_prevalence()
        #print(len(_parts_P[1]),len(_parts_P[2]),len(_parts_P[3]),len(_parts_P[4]))
        _kld_P=self.count_splited_KLD(_parts_P, _prev_test, _prev_test_estimate)
        print('\t\t\t\t VLP \t\t\t\t LP \t\t\t\t HP \t\t\t\t VHP \t\t\t\t total \n', _kld_P)
        _kld_D=self.count_splited_KLD(_parts_D, _prev_test, _prev_test_estimate)
        print('\t\t\t\t VLD \t\t\t\t LD \t\t\t\t HD \t\t\t\t VHD \t\t\t\t total \n', _kld_D)
        return 0

    def unite_cl_prob(self):
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

    def estimate_cl_prob(self):
        #[_csr, _y, y_names]=self._read_pickle(self._train_file)
        #_prev_train=self.count_prevalence(_y)
        #model=self.arff.fit(_csr,_y)
        #with open('texts/ml_model_'+self.dir_name+'.pickle', 'wb') as f:
        #    pickle.dump(model, f)
        #    f.close()
        with open('texts/ml_model_'+self.dir_name+'.pickle', 'rb') as f:
            model = pickle.load(f)
            f.close()
        _prob_list=[]
        _y1_list=[]
        for _t in range(42,52):#len(self._test_files)):
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

    def prob_classify_and_count(self, _indexes):
        [_y_train, _y_test_list, _pred_prob_list, _test_files, y_names]=_indexes
        avr_prob=[]
        for _pred_prob in _pred_prob_list:
            _prob=_pred_prob.T
            for cl_row in _prob:
                avr_prob.append(np.average(cl_row))
        return avr_prob

    def expectation_maximization(self, _indexes):
        [_y_train, _y_test_list, _pred_prob_list, _test_files, y_names]=_indexes
        #print(_pred_prob_list[0][1])
        pr_train=self.bin_prevalence(_y_train)
        pr_all=[]
        test_prevalence=[]
        num_iter=[]
        _test_num=0#0..3 len(_y_test_list)
        for _pred_prob in _pred_prob_list:# Test files loop
            print('Test file N', _test_num)
            test_prevalence=np.concatenate((test_prevalence,self.bin_prevalence(_y_test_list[_test_num])), axis=1)
            pr_c=[x for x in pr_train]

            _prob=_pred_prob.T
            for cl_n in range(len(pr_train)):#Category
                print('Test file N %s, class number %s' %(_test_num, cl_n))
                iter=0
                _delta=1
                while _delta>0.01:#0.00000000000000001:
                    pr_c_x=[]
                    _j=0
                    for pr_c_xk in _prob[cl_n]:#xk in category
                #Step E
                        pr_c_x_k=(pr_c[cl_n]/pr_train[cl_n]*pr_c_xk)/(((1-pr_c[cl_n])/(1-pr_train[cl_n]))*(1-pr_c_xk)+pr_c[cl_n]/pr_train[cl_n]*pr_c_xk)
                        #if _j==0: print(pr_c[_class_num],pr_train[_class_num], pr_c_xk, pr_c_x_k)
                        pr_c_x.append(pr_c_x_k)
                        _j+=1
                #Step M
                    pr_c_new=np.average(pr_c_x)#np.average(_prob[cl_n])
                    _delta=np.abs(pr_c_new-pr_c[cl_n])
                    #print(pr_c[cl_n],pr_c_new, test_prevalence[cl_n*(_test_num+1)], _delta)
                    #pr_train[cl_n]=pr_c[cl_n]
                    #_prob[cl_n]=pr_c_x_k
                    pr_c[cl_n]=pr_c_new
                    iter+= 1
                num_iter.append(iter)
            pr_all=np.concatenate((pr_all,pr_c), axis=1)
            _test_num+=1
        #for _j in range(len(test_prevalence)):
        #    print(test_prevalence[_j], pr_all[_j], pr_train[(_j % 88)],  _j)
        print(num_iter)
        return pr_all ,num_iter

    def __conditional_probability(self,p1,p2,val1,val2):
        _c=0
        for _i in range(len(p1)):
            if p1[_i]==val1 and p2[_i]==val2:
                _c=_c+1
        return _c/len(p1)

    def _kfold_tp_fp(self, n_folds=2):
    #return tru positive rate and false positive rate arrays
        [_csr, _y, y_names]=self._read_pickle(self._train_file)
        _kf=KFold(_y.shape[0],n_folds=n_folds)
        tp=[]
        fp=[]
        for train_index, test_index in _kf:
            X_train, X_test = csr_matrix(_csr.toarray()[train_index]), csr_matrix(_csr.toarray()[test_index])
            y_train, y_test = csr_matrix(_y.toarray()[train_index]), csr_matrix(_y.toarray()[test_index])
            model=self.arff.fit(X_train, y_train)
            y_predict=model.predict(X_test)
            tp_k=[]
            fp_k=[]
            for _i in range(y_train.toarray().shape[1]):
                s_true=y_test.toarray().T[_i]
                s_pred=y_predict.toarray().T[_i]
                #cm=metrics.confusion_matrix(s_true, s_pred)
                tp_k.append(self.__conditional_probability(s_pred, s_true, 1., 1.))#cm[0,0]/len(s_true))
                #try:
                fp_k.append(self.__conditional_probability(s_pred, s_true, 1., 0.))#cm[1,0]/len(s_true))#len(s_true))
                #except:
                #    #print(_i, metrics.confusion_matrix(s_true, s_pred), s_true, s_pred)
                #    fp_k.append(0)
            tp.append(tp_k)
            fp.append(fp_k)
        tp_av=np.asarray([np.average(tp_k) for tp_k in np.asarray(tp).T])
        fp_av=np.asarray([np.average(fp_k) for fp_k in np.asarray(fp).T])
        print('tp_av',tp_av)
        print('fp_av',fp_av)
        with open(self.prefix+self.dir_name+'_kFCV.pickle', 'wb') as f:
            pickle.dump([tp_av, fp_av], f)
        return [tp_av, fp_av]

    def adj_classify_and_count(self, _indexes):
        [_y_train, _y_test_list, _y_pred_list, _test_files, y_names]=_indexes
        try:
            with open(self.prefix+self.dir_name+'_kFCV.pickle', 'rb') as f:
                [tp_av, fp_av] = pickle.load(f)
        except:
            [tp_av, fp_av]=self._kfold_tp_fp(10)
        #[tp_av, fp_av]=self._kfold_tp_fp(10)
        pred_all=[]
        test_all=[]
        j=0
        for _y_pred in _y_pred_list:
            #print('pr, tp_av, fp_av',self.bin_prevalence(_y_pred), tp_av, fp_av)
            pred=(self.bin_prevalence(_y_pred)-fp_av)/(tp_av-fp_av)
            pred_all=np.concatenate((pred_all, pred), axis=1)
            test_all=np.concatenate((test_all, self.bin_prevalence(_y_test_list[j])), axis=1)
            j+=1
        print('test_all',test_all)
        print('pred_all',pred_all)
        return pred_all

name='QuantOHSUMED' #QuantRCV1 #QuantOHSUMED
q=Quantification(name)
#indexes=q.estimate_cl_indexes()
indexes=q._read_pickle('texts/cl_indexes_'+name+'.pickle')
td=q.classify_and_count(indexes[1])
#ed1=q.classify_and_count(indexes[2])
#ed2=q.adj_classify_and_count(indexes)

#q.estimate_cl_prob()
#q.unite_cl_prob()
prob=q._read_pickle('texts/cl_prob_'+name+'.pickle')

#ed3=q.classify_and_count(prob[2], is_prob=True)
#ed4=q.prob_classify_and_count(prob)
ed5, num_iter=q.expectation_maximization(prob)

q.count_diff1(td,ed5, num_iter)