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

def generate_data():
    #X, y, p_c, p_w_c = make_ml_data(n_samples=1000, n_features=10,n_classes=3, n_labels=1,length=50, allow_unlabeled=False, return_indicator=True,return_distributions=True, sparse=True)
    X,y=make_classification(n_samples=1000, n_features=10, n_informative=3, n_classes=3, n_clusters_per_class=2)
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
    ed=q._exp_max(y_train, pred_prob_list, stop_delta=0.5)
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
    return KLD
k=0
X,y=generate_data()
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.75)
EM1=quantify(X_train, X_test, y_train, y_test,'EM1')
PCC=quantify(X_train, X_test, y_train, y_test,'PCC')
print(EM1, PCC)
#quantify_OHSUMED()