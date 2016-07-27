# coding: utf-8
__author__ = 'Nikolay Karpov'

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

class text_processing():
    def __init__(self, n):
        self.__tok=TfidfVectorizer().build_tokenizer()
        self.tfidf_ngrams=TfidfVectorizer(tokenizer=self.__tokenize, preprocessor=self.__preprocessor,ngram_range=(1,n),
                                     analyzer="word" ,binary=False, stop_words='english', lowercase=False)

    def __tokenize(self, text):
        if text.lower()!='not available\n':
            lemms=[]
            wnl = WordNetLemmatizer()
            #st = PorterStemmer()
            for item in self.__tok(text):
                if item.isalpha():
                    lemms.append(wnl.lemmatize(item.lower()))
                    #lemms.append(st.stem(item.lower()))
                else:
                    if item.isdigit():
                        if int(item)>=1700 and int(item)<=2100:
                            lemms.append('YEAR')
                        else:
                            lemms.append('DIGIT')
                    else:
                        #pass
                        lemms.append(item)
                        if item[-2:]=='th' and item[:-2].isdigit() or item[-2:]=='st' and item[:-2].isdigit() or item[-2:]=='nd'and item[:-2].isdigit() or item[-2:]=='rd'and item[:-2].isdigit():
                            lemms.append('ORDERNUM')
                        elif item[-2:]=='pm' and item[:-2].isdigit() or item[-2:]=='am' and item[:-2].isdigit():
                            lemms.append('HOUR')
                        elif item=='4EXCL' or item=='5QUEST' or item=='6POINT':
                            lemms.append(item)
                        else:
                            lemms.append('NAME_NAME')
                            #print(item)
        else:
            lemms=[]
        #print(lemms)
        return lemms

    def __preprocessor(self, text):
        patterns = [
#link
                ("https?:\/\/\S*"," HTTPLNK "),
#                ("@[\w_-]+", " SOME_USER "),
#smile
                (":\)*\)", " SMILELOO "),
                (";\)*\)","  SMILELOO "),
                (";-\)*\)"," SMILELOO "),
                (":\]*\]"," SMILELOO "),
                ("=D[^a-z^1-9$]"," SMILELOO "),
                (";D[^a-z^1-9^$]"," SMILELOO "),
                (":D[^a-z^1-9^$]"," SMILELOO "),
                (":-D[^a-z^1-9^$]"," SMILELOO "),
#frown
                (":\(*\(","  FROWNLO "),
                (":-\(*\(","  FROWNLO "),
#several exclamations, quetsions, points
                ("\!*\!"," 4EXCL "),
                ("\?*\?"," 5QUEST "),
                ("\.*\."," 3POINT "),
#negation
                ("n't","n not")
        ]
        for pat, sub in patterns:
            text=re.sub(pat, sub, text) #замена последовательная, порядок в паттерне имеет значение
        return text

    def fit_transform(self, texts):

        return self.tfidf_ngrams.fit_transform(texts)

    def transform(self, raw_documents):
        return self.tfidf_ngrams.transform(raw_documents)

class char_processing():
    def __init__(self):
        self.__tok=TfidfVectorizer().build_tokenizer()
        self.tfidf_ngrams=TfidfVectorizer(preprocessor=self.__preprocessor,ngram_range=(3,5),
                                     analyzer="char" ,binary=False, stop_words='english', lowercase=True)
    def __preprocessor(self, text):
        patterns = [
#link
                ("https?:\/\/\S*"," HTTPLNK "),
#                ("@[\w_-]+", " SOME_USER "),
#smile
                (":\)*\)", " SMILELOO "),
                (";\)*\)","  SMILELOO "),
                (";-\)*\)"," SMILELOO "),
                (":\]*\]"," SMILELOO "),
                ("=D[^a-z^1-9$]"," SMILELOO "),
                (";D[^a-z^1-9^$]"," SMILELOO "),
                (":D[^a-z^1-9^$]"," SMILELOO "),
                (":-D[^a-z^1-9^$]"," SMILELOO "),
#frown
                (":\(*\(","  FROWNLO "),
                (":-\(*\(","  FROWNLO "),
#several exclamations, quetsions, points
                ("\!*\!"," 4EXCL "),
                ("\?*\?"," 5QUEST "),
                ("\.*\."," 3POINT "),
#negation
                ("n't","n not")
        ]
        for pat, sub in patterns:
            text=re.sub(pat, sub, text) #замена последовательная, порядок в паттерне имеет значение
        return text

    def fit_transform(self, texts):
        return self.tfidf_ngrams.fit_transform(texts)

    def transform(self, raw_documents):
        return self.tfidf_ngrams.transform(raw_documents)
