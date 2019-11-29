#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:14:36 2017

@author: pedroa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#from sklearn.grid_search import GridSearchCV
from sklearn import model_selection
from sklearn import svm
from sklearn import preprocessing

# Cargar el dataset
data = pd.read_csv('dataset3.csv',header=None)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


scaler = preprocessing.StandardScaler ()
X = scaler.fit_transform (X , y )
# Partimos el conjunto en train y test
x_train , x_test , y_train , y_test = model_selection.train_test_split(X , y , stratify =y , test_size =0.25 , random_state =256)

Cs = np.logspace(-5, 15, num=11, base=2)
Gs = np.logspace(-15, 3, num=9, base=2)
optimo = GridSearchCV(estimator=svm_model, param_grid=dict(C=Cs,gamma=Gs), n_jobs=-1,cv=5)
optimo.fit(X_train,y_train)
print (optimo.score(X_test,y_test))

# Entrenar el modelo SVM
#svm_model = svm.SVC(kernel='rbf',C=2, gamma=2)
#svm_model.fit(x_train, y_train)

#Para obtener el CCR (pregunta 3).
print("CCR: " ,svm_model.score(x_train,y_train))

# Representar los puntos
plt.figure(1)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

# Representar el hiperplano separador
plt.axis('tight')
# Extraer límites
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()

# Crear un grid con todos los puntos y obtener el valor Z devuelto por la SVM
XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
Z = svm_model.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Hacer un plot a color con los resultados
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

plt.show()
