# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 08:17:17 2020

@author: saurav.a.chakraborty
"""

#churn modelling 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Users\\saurav.a.chakraborty\\Documents\\Saurav\\practice\\kaggle\\Churn Modelling\\Churn_Modelling.csv')

X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#encoding categorical values
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1]