# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from nltk import WordNetLemmatizer
import pymorphy2
import codecs

# Should be added all characters
punctuation_marks = ['.',',', ';',':']

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

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

morph = pymorphy2.MorphAnalyzer()

# first_article = u'Это о технологиях и науке. Тут будет много предложений'
# second_article = u'Это еще одни предложения для сравнения. Что-то нужно бы сделали еще бы'
first = codecs.open('forbes_ru.txt', 'r',encoding = 'UTF-8')
second = codecs.open('mens_health_ru.txt', 'r', encoding = 'UTF-8')
first_article = fileToText(first)
second_article = fileToText(second)

print(first_article)
print second_article
en_article = 'Hello, my name is Kirill. I live in NN'

first_list_bow = toNormalForm(first_article)
second_list_bow = toNormalForm(second_article)

list_for_comparison = first_list_bow + second_list_bow

train_data_features = vectorizer.fit_transform(list_for_comparison)
train_data_features = train_data_features.toarray()

print(train_data_features)

