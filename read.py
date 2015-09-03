# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
# from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import pymorphy2
import codecs


# Should be added all characters
punctuation_marks = ['.',',', ';',':', '(', ')']
# Stopwords
stop_words = stopwords.words('russian')

train = ['Forbes','Forbes','Forbes', 'Mens-Health', 'Mens-Health', 'Mens-Health']

# Function of normalization
def toNormalForm(list):
    array_to_return = []

    for symbol in punctuation_marks:
        list = list.replace(symbol,'')
    bag_of_words_preparation = (list.split())

    for word in bag_of_words_preparation:
        p = morph.parse(word)[0]
        array_to_return.append(p.normal_form)

    return [' '.join(array_to_return)]
# End of function

# Reading data from file
def fileToText(file):
    article = ''
    for line in file:
        article = article + line
    return article
# End of function

vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = stop_words,
                             max_features = 5000)

morph = pymorphy2.MorphAnalyzer()

# first_article = u'Это о технологиях и науке. Тут будет много предложений'
# second_article = u'Это еще одни предложения для сравнения. Что-то нужно бы сделали еще бы'
train_files = ['forbes_ru.txt','f2.txt','f3.txt','mens_health_ru.txt','m2.txt','m3.txt']
test_files = ['f_test.txt','m_test.txt']
path='texts\\'

articles = []
for name in train_files:
    articles.append(fileToText(codecs.open( path+name, 'r',encoding = 'UTF-8')))

list_for_comparison = []
for article in articles:
    list_for_comparison += toNormalForm(article)

train_data_features = vectorizer.fit_transform(list_for_comparison)
train_data_features = train_data_features.toarray()

# Random Forest
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train)

# можно сделать функцией
articles = []
for name in test_files:
    articles.append(fileToText(codecs.open(path+name, 'r',encoding = 'UTF-8')))

list_for_comparison = []
for article in articles:
    list_for_comparison += toNormalForm(article)

test_data_features = vectorizer.transform(list_for_comparison)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

print(result)




