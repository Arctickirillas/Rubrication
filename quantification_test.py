# coding: utf-8
__author__ = 'Nikolay Karpov'

from sklearn.cross_validation import KFold
from quantification import Quantification
from sklearn.datasets import make_multilabel_classification as make_ml_data
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def generate_data():
    X, Y, p_c, p_w_c = make_ml_data(n_samples=150, n_features=20,n_classes=6, n_labels=1,length=50, allow_unlabeled=False, return_indicator=True,return_distributions=True, sparse=True)
    #kf=KFold(Y.shape[0],n_folds=1)
    #for train_index, test_index in kf:
    #    pass
    print(p_c)

def quantify_OHSUMED():
    q=Quantification(method='test', dir_name='QuantOHSUMED') #QuantRCV1 #QuantOHSUMED
    prob=q._read_pickle('texts/cl_prob_'+q.dir_name+'.pickle')
    indexes=q._read_pickle('texts/cl_indexes_'+q.dir_name+'.pickle')
    td=q._classify_and_count(prob[1])
    ed=q._expectation_maximization(prob)
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

def quantify(X,y):
    q=Quantification(method='CC')
    q.fit(X,y)
    prev_pred=q.predict(X)
    print('prev_pred',prev_pred)
    avg_KLD=q.score([X], [y])
    print('avg_KLD',avg_KLD)
    return avg_KLD

X,y=prepocess()
quantify(X,y)
#print(quantify_OHSUMED())