#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 21:15:48 2020

@author: carlos
"""

#Regresion Lineal Multiple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 
dataset = pd.read_csv('startup.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values 


#2 Codificar datos categoricos y dummies (0,1,2,..,etc.)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

Labelencoder_x = LabelEncoder()
x[:, 3] = Labelencoder_x.fit_transform(x[:,3])

ct = ColumnTransformer([("State", OneHotEncoder(),[3])], remainder = 'passthrough')
x = ct.fit_transform(x) 

# Evitar la trampa de las variables ficticias

x = x[:, 1:]



#3 Dividir dataset ( entrenamiento y test)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


#3 Escalado de variables (escalado: y normalizado)
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

# Ajustar el modelo de Regresion Lineal multiple con el cojunto de entranamiento.

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

#Prediccion de los resultados en el conjunto de testing
y_pred = regression.predict(x_test)

#Contruir el modelo optimo de RLM utilizando la Eliminacion hacia atras.
import statsmodels.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis=1)
x_opt = np.array (x [:, [0, 1, 2, 3, 4, 5]], dtype = float)

SL = 0.05
regression_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regression_OLS.summary()





