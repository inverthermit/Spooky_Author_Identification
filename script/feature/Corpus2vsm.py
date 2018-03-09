import copy
import random
import numpy as np
import math
import json
import os.path
import sys
sys.path.append('../')
from types import SimpleNamespace as Namespace
from util.Util import Util
import pandas as pd



sys.path.append('../')
rootDir = '../../'
csv_data = pd.read_csv(rootDir+Util.getConfig('train_csv'))#converted to utf-8
# print(csv_data)
train_corpus = csv_data[['text','author']].as_matrix()[:,0]
train_y = csv_data[['text','author']].as_matrix()[:,1]
# corpus = [x.encode('utf-8') for x in corpus]
print(train_corpus)



from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),stop_words = stopwords)#58%
# vectorizer = CountVectorizer()#54%

X = vectorizer.fit_transform(train_corpus)
features = vectorizer.get_feature_names()
print(len(features))
# print(X.toarray())


"""Feature selection"""
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.99 * (1 - .99)))
sel.fit(X)
X= sel.transform(X)
print(len(X.toarray()[0]))
print(X.toarray()[0])

from sklearn.neural_network import MLPClassifier
cls = MLPClassifier(max_iter = 500)

# from sklearn.ensemble import RandomForestClassifier
# cls = RandomForestClassifier(n_estimators=10)
cls.fit(X,train_y)
print(cls.score(X,train_y))


#
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_pred = cross_val_predict(cls,X,train_y,cv=10)
conf_mat = confusion_matrix(train_y,y_pred)
print(conf_mat)


from sklearn.metrics import classification_report
target_names = ['a', 'b','c']
print(classification_report(train_y,y_pred, target_names=target_names))
