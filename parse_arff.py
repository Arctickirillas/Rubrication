# coding: utf-8
__author__ = 'Nikolay Karpov'

import pyparsing as p
import os
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier as mc
from sklearn.linear_model import SGDClassifier as sgdc
from sklearn.preprocessing import MultiLabelBinarizer as mb
from scipy.sparse import csr_matrix

class parse_arff:
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

    def make_csr(self, _input, _feature_num=11286): #11286 21609
        _indptr = [0]
        _indices = []
        _data = []
        _data_names=[]
        _classes=[]
        for _element in _input.dataList:
            _indptr.append(len(_element)-2)
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
            _classes.append(_class)
        _classes_bin=mb().fit_transform(_classes)
        return csr_matrix((_data, _indices, _indptr),shape=[len(_classes), _feature_num],dtype=float), _classes_bin#.toarray()

    def fit(self, _input_X, _input_y):
        classif = mc(sgdc())
        model=classif.fit(_input_X, _input_y)
        return model

    def read_dir(self, _path):
        _files_test=[]
        _file_train=''
        _list_dir=os.listdir(_path)
        for _file in _list_dir:
            if _file.find('train')>0:
                _file_train=_path+_file
            elif _file.find('test')>0:
                _files_test.append(_path+_file)
        print(_file_train, _files_test)
        return _file_train, _files_test

    def execQuantRCV1(self):
        train_file, test_files=self.read_dir('QuantRCV1/')# QuantOHSUMED 11286# QuantRCV1 21610
        arff=self.read_arff(train_file)
        csr, y=self.make_csr(arff, 21610)
        model=self.fit(csr,y)

        arff1=self.read_arff(test_files[0])
        csr1, y1=self.make_csr(arff1, 21610)
        pr= model.predict(csr1)
        for st in pr:
            print(st)

    def execQuantOHSUMED(self):
        train_file, test_files=self.read_dir('QuantOHSUMED/')# QuantOHSUMED 11286# QuantRCV1 21610
        arff=self.read_arff(train_file)
        csr, y=self.make_csr(arff, 11286)
        model=self.fit(csr,y)

        arff1=self.read_arff(test_files[0])
        csr1, y1=self.make_csr(arff1, 11286)
        pr= model.predict(csr1)
        for st in pr:
            print(st)

pa=parse_arff()
pa.execQuantOHSUMED()