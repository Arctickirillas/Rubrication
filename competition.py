# coding: utf-8
__author__ = 'Kirill Rudakov'

import read as r
import quantify as q
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn import cross_validation
# from quantification import Quantification
import subprocess
import numpy as np
from time import sleep


def obtainSVMperfTwoPoint(sentiment,tweets,name = 'noname'):
    path = 'SemEval/data/two-point/'
    file = open(path+str(name)+'.txt','w')
    for i in range(len(sentiment)):
        if sentiment[i] == 'positive':
            file.write('1 ')
            for j,tw in enumerate(tweets[i]):
                if tw != 0.0:
                    file.write(str(j+1)+':'+str(tw)+' ')
            file.write('\n')
        elif sentiment[i] == 'negative':
            file.write('-1 ')
            for j,tw in enumerate(tweets[i]):
                if tw != 0.0:
                    file.write(str(j+1)+':'+str(tw)+' ')
            file.write('\n')
        else:
            file.write('0 ')
            for j,tw in enumerate(tweets[i]):
                if tw != 0.0:
                    file.write(str(j+1)+':'+str(tw)+' ')
            file.write('\n')
    file.close()

def obtainSVMperfFivePoint(sentiment,tweets,name = 'noname'):
    path = 'SemEval/data/five-point/'
    for m in range(-2,3,1):
        file = open(path+str(name)+'('+str(m)+')'+'.txt','w')
        for i in range(len(sentiment)):
            if str(sentiment[i]) == str(m):
                file.write('1 ')
                for j,tw in enumerate(tweets[i]):
                    if tw != 0.0:
                        file.write(str(j+1)+':'+str(tw)+' ')
                file.write('\n')
            elif str(sentiment[i]) == 'UNKNOWN':
                file.write('0 ')
                for j,tw in enumerate(tweets[i]):
                    if tw != 0.0:
                        file.write(str(j+1)+':'+str(tw)+' ')
                file.write('\n')
            else:
                file.write('-1 ')
                for j,tw in enumerate(tweets[i]):
                    if tw != 0.0:
                        file.write(str(j+1)+':'+str(tw)+' ')
                file.write('\n')
        file.close()


def SVMperfForTwoPoint(train=None,test=None,model=None,predictions=None):
    train = "SemEval/data/two-point/train.txt"
    test = "SemEval/data/two-point/test.txt"
    model = "SemEval/SVMperf/two-point/model.txt"
    predictions = "SemEval/SVMperf/two-point/predictions.txt"
    subprocess.Popen(["svm-perf-original/svm_perf_learn","-c","20",train,model], stdout=subprocess.PIPE)
    sleep(1)
    predict = subprocess.Popen(["svm-perf-original/svm_perf_classify",test,model,predictions], stdout=subprocess.PIPE)
    sleep(1)
    return predict.communicate()[0]

# Проблема с моделями!!! Нужно несколько раз запускать
def SVMperfForFivePoint(type=None):
    for i in range(-2,3,1):
        train = "SemEval/data/five-point/train("+str(i)+').txt'
        model =  "SemEval/SVMperf/five-point/model("+str(i)+").txt"
        subprocess.Popen(["svm-perf-original/svm_perf_learn","-c","20",train,model], stdout=subprocess.PIPE)
        sleep(1)
        for j in range(-2,3,1):
            test = "SemEval/data/five-point/test("+str(j)+").txt"
            predictions = "SemEval/SVMperf/five-point/predictions("+str(i)+")_("+str(j)+").txt"
            predict = subprocess.Popen(["svm-perf-original/svm_perf_classify",test,model,predictions], stdout=subprocess.PIPE)
            sleep(1)
            file = open('SemEval/SVMperf/five-point/report('+str(i)+")_("+str(j)+").txt" ,'wr')
            file.write(predict.communicate()[0])

def SVMperf():
    learn = subprocess.Popen(["svm-perf-original/svm_perf_learn","-c","20","SemEval/train.txt","SemEval/model.txt"], stdout=subprocess.PIPE)
    sleep(1)
    predict = subprocess.Popen(["svm-perf-original/svm_perf_classify","SemEval/test.txt","SemEval/model.txt","SemEval/predictions.txt"], stdout=subprocess.PIPE)
    sleep(1)
    print predict.communicate()[0]


def obtainPredictedAndTweets(tweetData):
    _predicted = []
    _tweets = []
    _topic = []
    file = open(tweetData, 'r')
    for line in file:
        data = line.split('\t')
        if data[3] != 'Not Available\n':
            _tweets.append(data[3])
            _predicted.append(data[2])
            _topic.append(data[1])
    return _predicted, _tweets, _topic

def kld(p, q):
        """Kullback-Leibler divergence D(P || Q) for discrete distributions when Q is used to approximate P
        Parameters
        p, q : array-like, dtype=float, shape=n
        Discrete probability distributions."""
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        return np.sum(np.where(p != 0,p * np.log(p / q), 0))

def getPredictions(predictions):
        q = []
        f = open(predictions,'r')
        for line in f:
            if float(line) >= 0.61 :
                q.append(1)
            elif float(line) < 0.61:
                q.append(-1)
        f.close()
        return  np.array(q)

def getPredictionsOption(predictions, option):
        q = []
        f = open(predictions,'r')
        for line in f:
            if float(line) >= option :
                q.append(1)
            elif float(line) < option:
                q.append(-1)
        f.close()
        return  np.array(q)

def getRealValue(data_test):
    q = []
    f = open(data_test,'r')
    for line in f:
        q.append(int(line.split(' ')[0]))
    f.close()
    return np.array(q)






# MAIN
# stop = stopwords.words('english')

predicted = []
tweets = []
topic = []

predicted,tweets, topic = obtainPredictedAndTweets('TweetsDownloaded/five-point/downloadedTrain.tsv')
predicted.reverse()

predictedTest, tweetsTest, topicTest = obtainPredictedAndTweets('TweetsDownloaded/testData/test.txt')


# http://www.nltk.org/api/nltk.tokenize.html
tknzr = TweetTokenizer()
stemmer = SnowballStemmer("english")
vectorizer = TfidfVectorizer(analyzer = "word",
                           tokenizer = None,
                           preprocessor = None,
                           stop_words = 'english'
                            )


tw = []
for t,statement in enumerate(tweets):
    tw.append(' '.join(stemmer.stem(i) for i in tknzr.tokenize(statement)))

twt = []
for t,statement in enumerate(tweetsTest):
    twt.append(' '.join(stemmer.stem(i) for i in tknzr.tokenize(statement)))

train = tw
sentimentTrain = predicted
test = twt
sentimentTest = predictedTest


train, test, sentimentTrain, sentimentTest = cross_validation.train_test_split(tw, predicted, test_size=0.3, random_state=15)

train_data_features= vectorizer.fit_transform(train)
train_data_features = train_data_features.toarray()

test_data_features = vectorizer.transform(test)
test_data_features = test_data_features.toarray()




def obtainTwoPointData():
    obtainSVMperfTwoPoint(sentimentTrain,train_data_features,'train')
    obtainSVMperfTwoPoint(sentimentTest,test_data_features,'test')


def obtainFivePointdata():
    obtainSVMperfFivePoint(sentimentTrain,train_data_features,'train')
    obtainSVMperfFivePoint(sentimentTest,test_data_features,'test')


# obtainFivePointdata()
# SVMperfForFivePoint()

# p = q.sentimentTest()
# qq = getPredictions()



qq = []
f = open('SemEval/SVMperf/two-point/predictions.txt','r')
for line in f:
    if float(line)>=1 :
        qq.append(1)
    elif float(line)<1:
        qq.append(-1)

def getOutData(q, topic, name = 'out'):
    top = ''
    countOfTop = 1.
    neg = 0.
    pos = 0.
    file = open (str(name)+'.output','w')
    for i in range(len(q)):
        if top==topic[i]:
            if q[i]==1:
                pos += 1
            else:
                neg += 1
            countOfTop += 1
        else:
            if top!='':
                file.write(str(top)+'\t'+str(float(pos/countOfTop))+'\t'+str(float(neg/countOfTop)) + '\n')
            top = topic[i]
            pos = 0.
            neg =0.
            if q[i]==1:
                pos += 1
            else:
                neg += 1
            countOfTop = 1
    file.write(str(top)+'\t'+str(float(pos/countOfTop))+'\t'+str(float(neg/countOfTop)) + '\n')
    file.close()


# getOutData(qq,topicTest,'test_out')





# qq = getPredictions("SemEval/SVMperf/five-point/predictions(0)_(0).txt")
# qq = q.CCforDouble(qq)
# print kld(p,qq)
# option = 0.61 # the best two-point
# option = -0.25 # the best five-point

option = -0.2
p = getRealValue("SemEval/data/five-point/test(2).txt")
p = q.CCforDouble(p)
qq = getPredictionsOption("SemEval/SVMperf/five-point/predictions(2)_(2).txt",option)
qq = q.CCforDouble(qq)
print kld(p,qq)
option = -1.
_kld = 1.
def preFunc():
    for i in range(-2,3,1):
        print i
preFunc()
# while option<1.3:
#
#     qq = getPredictionsOption("SemEval/SVMperf/five-point/predictions(0)_(0).txt",option)
#     qq = q.CCforDouble(qq)
#
# # file = open('SemEval/SVMperf/five-point/report(0)_(0).txt','r')
# # for line in file:
# #     print line
#     if kld(p,qq)<_kld:
#
#         _kld = kld(p,qq)
#         print option,':',kld(p,qq)
#     option += 0.01
exit()

# print 'Random Forest'
# print 'Classification:'
# sentimentTestEst,sentimentTestProbability = r.randomForest(train_data_features,sentimentTrain,test_data_features)
# print(r.metrics.classification_report(sentimentTest,sentimentTestEst))
# print 'Quantification:'
# q.CCforDouble(sentimentTestEst)
# q.CCforDouble(sentimentTestEst,sentimentTestProbability)


# print 'simpleSVM'
# print 'Classification:'
sentimentTestEst,sentimentTestProbability = r.simpleSVM(train_data_features,sentimentTrain,test_data_features, 100)
# print(r.metrics.classification_report(sentimentTest,sentimentTestEst))
# print 'Quantification:'
qq = q.CCforDouble(sentimentTestEst)
qq_prob = q.CCforDouble(sentimentTestEst,sentimentTestProbability)
p = q.CCforDouble(sentimentTest)









#
print kld(p,qq)
print kld(p,qq_prob)



# print 'multiClassSVM'
# sentimentTestEst = r.multiClassSVM(train_data_features,sentimentTrain,test_data_features)
# print(r.metrics.classification_report(sentimentTest,sentimentTestEst))




qq = []
f = open('SemEval/predictions.txt','r')
for line in f:
    if float(line)>0 and float(line)<0.5 or float(line)>1 :
        qq.append(1)
    elif float(line)<0 and float(line)>-0.5 or float(line)<-1:
        qq.append(-1)
    else:
        qq.append(int(round(float(line))))

qq = q.CCforDouble(qq)

print qq

print kld(p,qq)