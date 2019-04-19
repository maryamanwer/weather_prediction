# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:42:34 2019

@author: Rizwan Ahmed
"""
# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt


# Importing the dataset
dataset = pd.read_csv('temp.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Date'] = dataset['Date'].map(dt.datetime.toordinal)

X = dataset.iloc[:, 4:5].values
y = dataset.iloc[:, 2:3].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor.fit(X_train,y_train)


# Predicting a new result
y_pred = regressor.predict(X_test)

y_pred = np.reshape(y_pred, (366, 1))
y_diff = y_pred - y_test
y_diff_max = max(y_diff)
y_diff_st = np.std(y_diff)


import numpy as np

arr = np.array([list(range(365+1))])

arr = 1826+arr
arr =np.reshape(arr,(366,1))
y_pred_arr = regressor.predict([[1845]])
