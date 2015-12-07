# coding: utf-8
__author__ = 'Nikolay Karpov'

from sklearn.cross_validation import KFold, StratifiedKFold, train_test_split
from quantification import Quantification
from sklearn.datasets import make_multilabel_classification as make_ml_data
from sklearn.datasets import make_classification
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import scipy
import random
def generate_data():
    #X, y, p_c, p_w_c = make_ml_data(n_samples=1000, n_features=10,n_classes=3, n_labels=1,length=50, allow_unlabeled=False, return_indicator=True,return_distributions=True, sparse=True)
    X,y=make_classification(n_samples=10000, n_features=10, n_informative=3, n_classes=2, n_clusters_per_class=2)
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

def prepocess():
    from  sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    vect=TfidfVectorizer().build_tokenizer()
    def tokenize(text):
        lemms=[]
        #stems = []
        wnl = WordNetLemmatizer()
        #st = PorterStemmer()
        for item in vect(text):
            lemms.append(wnl.lemmatize(item))
            #stems.append(st.stem(item))
        return lemms

    categories = ['alt.atheism','talk.religion.misc','comp.graphics', 'sci.space']
    dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
    labels = dataset.target
    tfidf_ngrams=TfidfVectorizer(tokenizer=tokenize, ngram_range=(1,1), analyzer="word" ,binary=False, stop_words='english')
    X = tfidf_ngrams.fit_transform(dataset.data)
    return X, labels

def quantify(X_train, X_test, y_train, y_test,method):
    q=Quantification(method=method)
    q.fit(X_train,y_train)
    KLD=q.score([X_test], [y_test])
    #print('Test',q._classify_and_count([y_test]))
    #print('Train',q._classify_and_count([y_train]))
    return KLD


def drift(X,y,proportion=0.5):
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


k=0
for i in range(1):
    X,y=generate_data()
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.75)
    X_test_d,y_test_d=drift(X_test,y_test,0.5)

    q=Quantification(method='')
    q.fit(X_train,y_train)
    print(i, q._classify_and_count([y_test_d]))
    EM=q.score([X_test_d], [y_test_d],method='EM')
    EM1=q.score([X_test_d], [y_test_d],method='EM1')
    PCC=q.score([X_test_d], [y_test_d], method='PCC')
    ACC=q.score([X_test_d], [y_test_d], method='ACC')
    if EM1<PCC: k+=1
    print('EM1',EM1,'EM',EM,'PCC',PCC,'ACC', ACC, 'k',k)
print(k)
#quantify_OHSUMED()
