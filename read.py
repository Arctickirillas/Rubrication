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

#
testF = open('test.txt', 'w')

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
        print list_for_comparison

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
    return random_forest.predict(test),random_forest.predict_proba(test)

# SVM For Multi Class
def multiClassSVM(trained,predicted,test):
    svm = mc(SVC(kernel='linear', probability=True)) # for multi classes
    svm = svm.fit(trained,predicted)
    return svm.predict(test)

def simpleSVM(trained,predicted,test,max_iter):
    svm = SVC(kernel='linear', probability=True, max_iter=max_iter) # for multi classes
    svm = svm.fit(trained,predicted)
    return svm.predict(test),svm.predict_proba(test)

# Obtaining Train Data From Dump
file = open('pickle_QuantOHSUMED/quant_OHSUMED_train.arff.pickle','rb')
pickle_file = p.load(file)
file.close()

# Splitting Into Trained, Predicted and Label
trained = pickle_file[0]
predicted = pickle_file[1]
class_name = pickle_file[2]

myPred = predicted.toarray()

# Train.dat
# for i in range(2510):
#             for j in range(1):
#                 if myPred[i][j] == 1:
#                     train.write('1 ')
#                     arrPr = trained[i].toarray()
#                     for m in range(len(arrPr[0])):
#                         if arrPr[0][m]!=0:
#                             train.write(str(m+1)+':'+str(arrPr[0][m])+' ')
#                     train.write('\n')
# for i in range(2510):
#             for j in range(1,88):
#                 if myPred[i][j] != 1:
#                     train.write('-1 ')
#                     arrPr = trained[i].toarray()
#                     for m in range(len(arrPr[0])):
#                         if arrPr[0][m]!=0:
#                             train.write(str(m+1)+':'+str(arrPr[0][m])+' ')
#                     train.write('\n')
#             print i


def obtainTrain(myPred):
    for j in range(88):
        train = open('dat/train'+str(j)+'.txt', 'w')
        for i in range(2510):
                if myPred[i][j] == 1:
                    train.write('1 ')
                    arrPr = trained[i].toarray()
                    for m in range(len(arrPr[0])):
                        if arrPr[0][m]!=0:
                            train.write(str(m+1)+':'+str(arrPr[0][m])+' ')
                    train.write('\n')
                else:
                    train.write('-1 ')
                    arrPr = trained[i].toarray()
                    for m in range(len(arrPr[0])):
                        if arrPr[0][m]!=0:
                            train.write(str(m+1)+':'+str(arrPr[0][m])+' ')
                    train.write('\n')
        print j
        train.close()




# MultiLabelBinarizer().fit_transform(pr)

# Obtaining Test Data From Dump
file = open('pickle_QuantOHSUMED/quant_OHSUMED_test_88.arff.pickle','rb')
pickle_file = p.load(file)
file.close()

test = pickle_file[0]
test_labels = pickle_file[1]

myPred = test_labels.toarray()
# Test.dat
# for i in range(3207):
#             for j in range(1):
#                 if myPred[i][j] == 1:
#                     testF.write('+1 ')
#                     arrPr = test[i].toarray()
#                     for m in range(len(arrPr[0])):
#                         if arrPr[0][m]!=0:
#                             testF.write(str(m+1)+':'+str(arrPr[0][m])+' ')
#                     testF.write('\n')
#
# for i in range(3207):
#             for j in range(1,88):
#                 if myPred[i][j] != 1:
#                     testF.write('-1 ')
#                     arrPr = test[i].toarray()
#                     for m in range(len(arrPr[0])):
#                         if arrPr[0][m]!=0:
#                             testF.write(str(m+1)+':'+str(arrPr[0][m])+' ')
#                     testF.write('\n')
#             print i


# result = svm.predict(test) #0 .85      0.57      0.66      6124

# predict_proba = svm.predict_proba(test)
# print(result[0])

# print(len(test_labels))
#
# print(metrics.classification_report(test_labels,result))


# examleOfRubrication()
# # print multiClassSVM(trained,predicted,test)
# # print  trained
#
# obtainTrain(myPred)