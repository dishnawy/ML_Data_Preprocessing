#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 20:20:24 2017

@author: Dish
"""
#Data processing
import pandas as pd

#Importing data
path_to_data = 'path/to/data.csv'

dataset = pd.read_csv(path_to_data)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:4])
X[:, 1:4] = imputer.transform(X[:, 1:4])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the data into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
