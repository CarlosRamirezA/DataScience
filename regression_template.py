#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:38:24 2020

@author: carlos
"""


# Plantilla de Regresion

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values 

"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
"""
"""
#3 Escalado de variables (escalado: y normalizado)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1))"""

# Ajustar la regresion con el dataset




# Prediccion de nuestros modelos  
y_pred = regression.predict()



# Visualizacion de los resultados del Modelo Polinomico
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color = "red")
plt.plot(x_grid, predict(x_grid, regression.predict(x_grid)), color = "blue")
plt.title("Modelo de Regresion")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()





