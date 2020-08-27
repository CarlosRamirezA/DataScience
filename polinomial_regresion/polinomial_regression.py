#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 18:22:19 2020

@author: carlos
"""


# Regresion polinomica


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

# Ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Ajustar la regresion polinomica lineal con el dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)  
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# visualizacion de los resultados del Modelo Lineal
plt.scatter(x, y, color = "red")
plt.plot(x, lin_reg.predict(x), color = "blue")
plt.title("Modelo de Regresion Lineal")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()




# Visualizacion de los resultados del Modelo Polinomico
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color = "red")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = "blue")
plt.title("Modelo de Regresion Polinomica")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Prediccion de nuestros modelos  

lin_reg_2.predict(poly_reg.fit_transform(x))



