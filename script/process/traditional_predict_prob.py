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



class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vectName = rootDir+Util.getConfig('tmp') + save_name +'.vectorizer'
vectorizer = joblib.load(vectName)
selName = rootDir+Util.getConfig('tmp') + save_name +'.vt'
sel = joblib.load(selName)
"""Generate test BOW"""
test_csv_data = pd.read_csv(rootDir+Util.getConfig('test_csv'))
test_corpus = test_csv_data[['text']].as_matrix()[:,0]
ids = test_csv_data[['id']].as_matrix()[:,0]
# test_corpus = [x.encode('utf8') for x in test_corpus]
# print(test_corpus)
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),stop_words = stopwords)#58%
# vectorizer = CountVectorizer()#54%

X = vectorizer.fit_transform(test_corpus)
test_X = vectorizer.transform(test_corpus)
test_X = sel.transform(test_X)
# print(len(test_X.toarray()[0]))
# print(test_X.toarray()[0])

modelName = rootDir+Util.getConfig('tmp') + save_name +'.pkl'
cls = joblib.load(modelName)
predict_test_y = cls.predict_proba(test_corpus)
print(predict_test_y)
# file = open(rootDir+Util.getConfig('tmp')+'testfile.txt','w')
# for name in predict_test_y:
#     file.write(name)
# file.close()
#
# modelName = rootDir+Util.getConfig('tmp') + 'ann500iter.pkl'
# from sklearn.externals import joblib
# cls = joblib.load(modelName)
