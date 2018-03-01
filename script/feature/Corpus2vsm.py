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
csv_data = pd.read_csv(rootDir+Util.getConfig('train_csv'))
print(csv_data)
corpus = csv_data[['text','author']].as_matrix()

from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer()
dataset = vectorizer.fit_transform(corpus[:,0])
print(dataset.toarray())

from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier()
cls.fit(dataset,corpus[:,1])
cls.score()

# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import confusion_matrix
# y_pred = cross_val_predict(clf,X,y,cv=10)
# conf_mat = confusion_matrix(y,y_pred)
# print(conf_mat)
#
# from sklearn.metrics import classification_report
# target_names = ['0', '1']
# print(classification_report(y,y_pred, target_names=target_names))
