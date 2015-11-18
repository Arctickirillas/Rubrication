__author__ = 'Kirill Rudakov'

import read as r
import quantify as q
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

stop = stopwords.words('english')

predicted = []
tweets = []

downloadedFile = open('downloaded.tsv', 'r')
for line in downloadedFile:
    data = line.split('\t')
    predicted.append(data[2])
    tweets.append(data[3])

predicted.reverse()
# print tweets

stemmer = SnowballStemmer("english")

# http://www.nltk.org/api/nltk.tokenize.html
tknzr = TweetTokenizer()

vectorizer = TfidfVectorizer(analyzer = "word",
                           tokenizer = None,
                           preprocessor = None,
                           stop_words = 'english'
                            )

tw = []
for t,statement in enumerate(tweets):
    tw.append(' '.join(stemmer.stem(i) for i in tknzr.tokenize(statement)))


train = tw[:len(tw)/2]
test = tw[len(tw)/2+1:]
sentimentTrain = predicted[:len(tw)/2]
sentimentTest = predicted[len(tw)/2+1:]

train_data_features= vectorizer.fit_transform(train)
train_data_features = train_data_features.toarray()

test_data_features = vectorizer.transform(test)
test_data_features = test_data_features.toarray()

print 'Random Forest'
print 'Classification:'
sentimentTestEst,sentimentTestProbability = r.randomForest(train_data_features,sentimentTrain,test_data_features)
print(r.metrics.classification_report(sentimentTest,sentimentTestEst))
print 'Quantification:'
q.CCforDouble(sentimentTestEst)
# q.CCforDouble(sentimentTestEst,sentimentTestProbability)

print 'simpleSVM'
print 'Classification:'
sentimentTestEst,sentimentTestProbability = r.simpleSVM(train_data_features,sentimentTrain,test_data_features, 100)
print(r.metrics.classification_report(sentimentTest,sentimentTestEst))
print 'Quantification:'
q.CCforDouble(sentimentTestEst)
q.CCforDouble(sentimentTestEst,sentimentTestProbability)

# print 'multiClassSVM'
# sentimentTestEst = r.multiClassSVM(train_data_features,sentimentTrain,test_data_features)
# print(r.metrics.classification_report(sentimentTest,sentimentTestEst))

