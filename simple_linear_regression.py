#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 22:07:07 2020

@author: carlos
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 
dataset = pd.read_csv('salary_data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values 



#3 Dividir dataset ( entrenamiento y test)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state = 0)

#3 Escalado de variables (escalado: y normalizado)
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

#Crear modelo de Regresion Lineal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regresion = LinearRegression()
regresion.fit(x_train, y_train)

#predecir el conjunto de test
y_pred = regresion.predict(x_test)

#Visualizar los resultados de entrenamiento
plt.scatter(x_train,y_train, color ="red")
plt.plot(x_train, regresion.predict(x_train), color="blue")
plt.title("sueldo vs anios de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Anios de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

#Visualizar los resultados de test
plt.scatter(x_test,y_test, color ="red")
plt.plot(x_train, regresion.predict(x_train), color="blue")
plt.title("sueldo vs anios de Experiencia (Conjunto de Testing)")
plt.xlabel("Anios de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()