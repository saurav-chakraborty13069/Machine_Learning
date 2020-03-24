# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:29:26 2018

@author: gourav chakraborty
"""

#Importing Libraries
import pandas as pd
import numpy as np
#import matplotlib as plt
#import seaborn as sbn
import pickle, requests, json

#Importing Dataset
dataset = pd.read_csv("./ML_Projects/flask/Iris/iris2.csv")

X = dataset.drop(['iris'],axis=1)
y = dataset['iris']

#Encoding categorical data of dependent variable

#Splitting dtaset in train test set
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size = 0.2)

#fitting the model to train set
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression()
logistic_classifier.fit(train_X,train_y)

#predicting the test set
y_pred = logistic_classifier.predict(test_X)

#calculating accuracy score
from sklearn import metrics
accuracy = metrics.accuracy_score(test_y, y_pred)
accuracy

#saving model to disk
pickle.dump(logistic_classifier, open('./ML_Projects/flask/Iris/model1.pkl','wb'))
