# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:10:43 2019

@author: saurav.a.chakraborty
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
from sklearn.externals import joblib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


train_data = pd.read_csv('C:\\Users\\saurav.a.chakraborty\\Documents\\Saurav\\practice\\AV\\HR Analytics\\train_LZdllcl.csv')
test_data = pd.read_csv('C:\\Users\\saurav.a.chakraborty\\Documents\\Saurav\\practice\\AV\\HR Analytics\\test_2umaH9m.csv')
test_data.head()

'''categorical variables-------
department
education
gender
recruitment_channel
'''

'''
to be changed to numeric--------
region
'''

'''
numeric---------
no_of_trainings
age
previous_year_rating
length_of_service
KPIs_met >80%
awards_won?
avg_training_score
'''

region = []
for i in range(0, len(train_data['region'])):
    region.append(int(train_data['region'][i][-1]))
train_data['region'] = region

region = []
for i in range(0, len(test_data['region'])):
    region.append(int(test_data['region'][i][-1]))
test_data['region'] = region


count = 0
for i in range(0, len(train_data['education'])):
    if str(train_data['education'][i]) == 'nan':
        #print("null value found")
        train_data['education'][i] = "Bachelor's"
          
count = 0
for i in range(0, len(test_data['education'])):
    if str(test_data['education'][i]) == 'nan':
        #print("null value found")
        test_data['education'][i] = "Bachelor's"
        
        
mean_rating = train_data['previous_year_rating'].mean()
for i in range(0, len(train_data['previous_year_rating'])):
    if str(train_data['previous_year_rating'][i]) == 'nan':
        #print("null value found")
        train_data['previous_year_rating'][i] = mean_rating
        
mean_rating = test_data['previous_year_rating'].mean()
for i in range(0, len(test_data['previous_year_rating'])):
    if str(test_data['previous_year_rating'][i]) == 'nan':
        #print("null value found")
        test_data['previous_year_rating'][i] = mean_rating
        
        
X = train_data.iloc[:,1:13 ].values
y = train_data.iloc[:, 13].values
X_test_data = test_data.iloc[:,1: ].values

labelencoder1 = LabelEncoder()
X[:,0] = labelencoder1.fit_transform(X[:,0])
labelencoder2 = LabelEncoder()
X[:,2] = labelencoder2.fit_transform(X[:,2])
labelencoder3 = LabelEncoder()
X[:,3] = labelencoder3.fit_transform(X[:,3])
labelencoder4 = LabelEncoder()
X[:,4] = labelencoder4.fit_transform(X[:,4])
onehotencoder1 = OneHotEncoder(categorical_features = [0, 2, 4])
X = onehotencoder1.fit_transform(X).toarray()
X = X[:, 3:]
X.shape
#dummy variable removal

labelencoder1 = LabelEncoder()
X_test_data[:,0] = labelencoder1.fit_transform(X_test_data[:,0])
labelencoder2 = LabelEncoder()
X_test_data[:,2] = labelencoder2.fit_transform(X_test_data[:,2])
labelencoder3 = LabelEncoder()
X_test_data[:,3] = labelencoder3.fit_transform(X_test_data[:,3])
labelencoder4 = LabelEncoder()
X_test_data[:,4] = labelencoder3.fit_transform(X_test_data[:,4])
onehotencoder1 = OneHotEncoder(categorical_features = [0, 2, 4])
X_test_data = onehotencoder1.fit_transform(X_test_data).toarray()
X_test_data.shape
X_test_data = X_test_data[:, 3:]
X_test_data.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#=============================================================================
classifier = Sequential()
classifier.add(Dense(units=6,  input_shape=(21,), activation='relu'))
#classifier.add(Dropout(0.2))
classifier.add(Dense(units=6,  activation='relu'))
#classifier.add(Dropout(0.2))
classifier.add(Dense(units=1,  activation='sigmoid'))
#classifier.add(Dropout(0.2))
classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#=============================================================================
y_pred = classifier.predict(X_test)
for i in range(0, len(y_pred)):
    if y_pred[i]<0.5:
        y_pred[i] = int(0)
    else:
        y_pred[i] = int(1)

#==============================================================================
test_loss, test_acc = classifier.evaluate(X_test, y_test)
print('test_acc:', test_acc)

cm = confusion_matrix(y_test, y_pred)
f_score = f1_score(y_test, y_pred)

#==============================================================================
pred = classifier.predict(X_test_data)
for i in range(0, len(pred)):
    if pred[i]<0.5:
        pred[i] = int(0)
    else:
        pred[i] = int(1)
        
#=============================================================================
'''
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6,  input_shape=(21,), activation='relu'))
    classifier.add(Dense(units=6,  activation='relu'))
    classifier.add(Dense(units=1,  activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
'''
#=============================================================================

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6,  input_shape=(21,), activation='relu'))
    classifier.add(Dense(units=6,  activation='relu'))
    classifier.add(Dense(units=1,  activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



#=============================================================================        
df = pd.DataFrame(columns = ['employee_id','is_promoted'])
        
df['employee_id'] = test_data['employee_id']
df['is_promoted'] = pred.astype('int64')


df.to_csv('C:\\Users\\saurav.a.chakraborty\\Documents\\Saurav\\practice\\AV\\HR Analytics\\sub6.csv', index = False)

classifier.save('model.h5')
joblib.dump(classifier, 'joblib_model.pkl') 
pickle.dump(classifier, open('pickle_model.pkl','wb'))



