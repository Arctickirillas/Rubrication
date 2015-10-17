# coding: utf-8
__author__ = 'Nikolay Karpov'

import pyparsing as p
import os, pickle
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier as mc
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier as SGDC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import MultiLabelBinarizer as mb
from sklearn import metrics
from scipy.sparse import csr_matrix

class Parse_ARFF:
    def __init__(self):
        pass

    def read_arff(self, _fname):
        text = ''.join(open(_fname, 'r').readlines())
        relationToken = p.Keyword('@RELATION', caseless=True)
        dataToken = p.Keyword('@DATA', caseless=True)
        attribToken = p.Keyword('@ATTRIBUTE', caseless=True)
        ident = p.ZeroOrMore(p.Suppress('\''))+p.Word( p.alphas, p.alphanums + '_-.' ).setName('identifier')+p.ZeroOrMore(p.Suppress('\''))
        relation = p.Suppress(relationToken) + p.ZeroOrMore(p.Suppress('"'))\
                   +ident.setResultsName('relation') + p.ZeroOrMore(p.Suppress('"'))
        attribute = p.Suppress(attribToken)+p.quotedString.setParseAction(lambda t: t.asList()[0].strip("'")).setResultsName('attrname') + p.Suppress(p.restOfLine)
        int_num = p.Word(p.nums)
        pm_sign = p.Optional(p.Suppress("+") | p.Literal("-"))
        float_num = p.Combine(pm_sign + int_num + p.Optional('.' + int_num) + p.Optional('e' + pm_sign + int_num)).setParseAction(lambda t: float(t.asList()[0]))
        module_name=p.Group((int_num.setParseAction(lambda t: int(t.asList()[0]))).setName('Key')+
                            (p.quotedString.setParseAction(lambda t: t.asList()[0].strip("'"))|float_num).setName('Value')+
                             p.Suppress(','))
        dataList = (p.Suppress('{')+p.OneOrMore(module_name)+p.Suppress('}')).setParseAction(lambda t: [t.asList()])
        comment = '%' + p.restOfLine
        arffFormat = (p.OneOrMore(p.Suppress(comment))+relation.setResultsName('relation') +
                       p.OneOrMore(attribute).setResultsName('identifiers')+
                       dataToken+
                       p.OneOrMore(dataList).setResultsName('dataList')
                      ).setResultsName('arffdata')
        tokens =  arffFormat.parseString(text)
        featureNames=tokens.arffdata.identifiers
        return (tokens.arffdata)

    def make_csr(self, _input, _feature_num=11286, _class_num=88): # QuantOHSUMED 11286 88 # QuantRCV1 21610 99
        _indptr = [0]
        _indices = []
        _data = []
        _data_names=[]
        _classes_bin=[]
        _i=0
        _a0=[0 for i in range(_class_num)]
        for _element in _input.dataList:
            _class=[]
            for _pair in _element:
                if _pair[0]==0:
                    _data_names.append(_pair[1])
                    __name=_pair[1]
                elif _pair[0]>_feature_num:
                    _class.append(_pair[0])
                else:
                    _indices.append(_pair[0]-1)
                    _data.append(_pair[1])
                    _i += 1
            _indptr.append(_i)
            #make matrix like _classes_bin=mb().fit_transform(_classes)
            _line=[0 for i in range(_class_num)]
            for _it in _class:
                _line[_it-_feature_num-1]=1
            _classes_bin.append(_line)
        #make class names
        _ident=_input.identifiers[_feature_num+1:]
        #make scr_matrix
        _scr=csr_matrix((_data, _indices, _indptr),shape=[len(_classes_bin), _feature_num],dtype=float)
        return [_scr, csr_matrix(_classes_bin), _ident]

    def make_binary(self, _input_y, _num=0):
        _y=[]
        _i=0
        for _line in _input_y:
            if _line[_num]==1:
                _y.append(1)
                _i += 1
            else:
                _y.append(-1)
        return _y

    def make_dat_file(self, _input_X, _y):
        _i=0
        for _index in _y:
            _str='%d' %_index
            #for _doc in _input_X.getrow(_i):
                #print(_doc)
            #print(_str)
            _i=_i+1
        return 0

    def read_dir(self, _path):
        _files_test=[]
        _file_train=''
        _list_dir=os.listdir(_path)
        for _file in _list_dir:
            if _file.find('train')>0:
                _file_train=_path+'/'+_file
            elif _file.find('test')>0:
                _files_test.append(_path+'/'+_file)
        #print(_file_train, _files_test)
        return _file_train, _files_test

    def fit(self, _input_X, _input_y):
        #classif=LR( )
        classif = mc(SVC(kernel='linear', probability=True))#rbf poly sigmoid
        model=classif.fit(_input_X, _input_y)
        return model

    def convert_arff(self, dir_name='QuantOHSUMED', is_predict=False):
            # Read ARFF files;
            # serialize data to pickle format;
            # learn ML model predict probabilities and serialize results to pickle format
        if dir_name=='QuantOHSUMED':# QuantOHSUMED num_of_feat=11286 num_of_classes=88
            num_of_feat=11286
            num_of_classes=88
        elif dir_name=='QuantRCV1':# QuantRCV1 num_of_feat=21610 num_of_classes=99
            num_of_feat=21610
            num_of_classes=99
        train_file, test_files=self.read_dir(dir_name)
        arff=self.read_arff(train_file)
        [csr, y, y_names]=self.make_csr(arff, num_of_feat, num_of_classes)
        with open('texts/pickle_'+train_file+'.pickle', 'wb') as f:
            pickle.dump([csr, y, y_names], f)
            f.close()
        if is_predict:
            model=self.fit(csr,y)
        prob_list=[]
        y1_list=[]
        for test_file in test_files:
            arff1=self.read_arff(test_file)
            [csr1, y1, y1_names]=self.make_csr(arff1, num_of_feat, num_of_classes)
            with open('texts/pickle_'+test_file+'.pickle', 'wb') as f:
                pickle.dump([csr1, y1, y1_names], f)
                f.close()
            print('texts/pickle_'+test_file+'.pickle')
            if is_predict:
                prob_y1= model.predict_proba(csr1)
                print(metrics.classification_report(y1,pr_y1))
                prob_list.append(prob_y1)
                y1_list.append(y1)
                with open('texts/cl_prob_'+test_file+'.cl_prob', 'wb') as f:
                    pickle.dump(prob_y1, f)
                    f.close()
        if is_predict:
            with open('texts/cl_prob_'+dir_name+'.pickle', 'wb') as f:
                pickle.dump([y, y1_list, prob_list, test_files, y_names], f)
                f.close()
        return 0
#pa=Parse_ARFF()
#pa.convert_arff('QuantOHSUMED') # 'QuantRCV1'