# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
#Plantilla de Pre procesado

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 
dataset = pd.read_excel('Book.xlsx')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values 



#3 Dividir dataset ( entrenamiento y test)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#3 Escalado de variables (escalado: y normalizado)
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""


