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
from sklearn.externals import joblib
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_selection import VarianceThreshold

sys.path.append('../')
rootDir = '../../'
save_name = 'ann500iter'
csv_data = pd.read_csv(rootDir+Util.getConfig('train_csv'))#converted to utf-8
# print(csv_data)
train_corpus = csv_data[['text','author']].as_matrix()[:,0]
train_y = csv_data[['text','author']].as_matrix()[:,1]
# corpus = [x.encode('utf-8') for x in corpus]
print(train_corpus)


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

stopwords = set(stopwords.words('english'))
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),stop_words = stopwords)#58%
# vectorizer = CountVectorizer()#54%

X = vectorizer.fit_transform(train_corpus)
features = vectorizer.get_feature_names()
print(len(features))
# print(X.toarray())


"""Feature selection"""

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
# print(cls.score(X,train_y))
model_name = rootDir+Util.getConfig('tmp') + save_name +'.pkl'
joblib.dump(cls, model_name)

vectName = rootDir+Util.getConfig('tmp') + save_name +'.vectorizer'
joblib.dump(vectorizer, vectName)

vtName = rootDir+Util.getConfig('tmp') + save_name +'.vt'
joblib.dump(vectorizer, vtName)
