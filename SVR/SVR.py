#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:06:52 2020

@author: carlos
"""


#SVR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values 

#2 Dividir dataset ( entrenamiento y test)
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
"""

#3 Escalado de variables (escalado: y normalizado)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1))

# Ajustar la regresion con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(x,y)



# Prediccion de nuestros modelos con SVR
y_pred = regression.predict(sc_x.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(y_pred)


# datos no escalados



# Visualizacion de los resultados del SVR
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color = "red")
plt.plot(x_grid,regression.predict(x_grid), color = "blue")
plt.title("Modelo de Regresion (SVR)")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()











