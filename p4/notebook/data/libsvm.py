#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:14:36 2017

@author: pedroa
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import preprocessing
import sklearn.model_selection


# Cargar el dataset
data = pd.read_csv('train_spam.csv',header=None)
X_train = data.iloc[:,:-1].values
y_train = data.iloc[:,-1].values
data = pd.read_csv('test_spam.csv',header=None)
X_test = data.iloc[:,:-1].values
y_test = data.iloc[:,-1].values

# Entrenar el modelo SVM
svm_model = svm.SVC(kernel='rbf',C=10e-2, gamma='scale')
svm_model.fit(X_train,y_train)

precisionTrain = svm_model.score(X_train, y_train)
precisionTest = svm_model.score(X_test, y_test)
print("CCR train = %.2f%% | CCR test = %.2f%%" %(precisionTrain*100, precisionTest*100))
prediction = svm_model.predict(X_test)
print(prediction)
print(y_test)
print("\n", end='')

matriz_confusion = confusion_matrix(y_test, prediction)
print(matriz_confusion)
print("\n", end='')
print(svm_model.score(X_test,y_test))
