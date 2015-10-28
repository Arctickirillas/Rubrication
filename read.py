# coding: utf-8

__author__ = 'Rudakov Kirill'

# IMPORT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import pymorphy2
import codecs
import pickle as p
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier as mc
from sklearn import metrics

# Load From Dump
# f=open('dump','rb')
# res = p.load(f)
# print res

# StopSymbols  and StopWords| Should be added all characters
punctuation_marks = ['.',',', ';',':', '(', ')']
stop_words = stopwords.words('russian')

# Example
def examleOfRubrication():

    def obtainingTrainDataFeatures(train_files,vectorizer):
        articles = []

        for name in train_files:
            articles.append(fileToText(codecs.open(path + name, 'r',encoding = 'UTF-8')))

        list_for_comparison = []
        for article in articles:
            list_for_comparison += toNormalForm(article)

        type_data_features = vectorizer.fit_transform(list_for_comparison)
        type_data_features = type_data_features.toarray()

        return type_data_features

    def obtainingTestDataFeatures(test_files,vectorizer):
        articles = []

        for name in test_files:
            articles.append(fileToText(codecs.open(path + name, 'r',encoding = 'UTF-8')))

        list_for_comparison = []
        for article in articles:
            list_for_comparison += toNormalForm(article)

        type_data_features = vectorizer.transform(list_for_comparison)
        type_data_features = type_data_features.toarray()

        return type_data_features

    path='texts/'
    train = [0, 0, 0, 1, 1, 1]
    train_files = ['forbes_ru.txt','f2.txt','f3.txt','mens_health_ru.txt','m2.txt','m3.txt']
    test_files = ['f_test.txt','m_test.txt']

    vectorizer = toVectorize()

    train_data_features = obtainingTrainDataFeatures(train_files,vectorizer)
    test_data_features = obtainingTestDataFeatures(test_files,vectorizer)

    print randomForest(train_data_features,train,test_data_features)

# Function Of Normalization
def toNormalForm(list):
    morph = pymorphy2.MorphAnalyzer()
    array_to_return = []

    for symbol in punctuation_marks:
        list = list.replace(symbol,'')
    bag_of_words_preparation = (list.split())

    for word in bag_of_words_preparation:
        p = morph.parse(word)[0]
        array_to_return.append(p.normal_form)

    return [' '.join(array_to_return)]

# Reading Data From File
def fileToText(file):
    article = ''
    for line in file:
        article = article + line
    return article

# Tfidf Representation
def toVectorize():
    return TfidfVectorizer(analyzer = "word",
                           tokenizer = None,
                           preprocessor = None,
                           stop_words = stop_words,
                           max_features = 5000)
    # Or BagOfWords
    # CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = stop_words,max_features = 5000)

# Random Forest
def randomForest(trained,predicted,test):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest = random_forest.fit(trained, predicted)
    return random_forest.predict(test)

# SVM For Multi Class
def multiClassSVM(trained,predicted,test):
    svm = mc(SVC(kernel='linear', probability=True)) # for multi classes
    svm = svm.fit(trained,predicted)
    return svm.predict(test)

# Obtaining Train Data From Dump
file = open('pickle_QuantOHSUMED/quant_OHSUMED_train.arff.pickle','rb')
pickle_file = p.load(file)
file.close()

# Splitting Into Trained, Predicted and Label
trained = pickle_file[0]
predicted = pickle_file[1]
class_name = pickle_file[2]

# MultiLabelBinarizer().fit_transform(pr)

# Obtaining Test Data From Dump
file = open('pickle_QuantOHSUMED/quant_OHSUMED_test_88.arff.pickle','rb')
pickle_file = p.load(file)
file.close()

test = pickle_file[0]
test_labels = pickle_file[1]

# result = svm.predict(test) #0 .85      0.57      0.66      6124

# predict_proba = svm.predict_proba(test)
# print(result[0])

# print(len(test_labels))
#
# print(metrics.classification_report(test_labels,result))


# examleOfRubrication()
print multiClassSVM(trained,predicted,test)
