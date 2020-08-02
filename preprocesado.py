# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 
dataset = pd.read_excel('Book.xlsx')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values 


#1. Tratamientos de NAN para determinadas columnas y sacando el promedio

from sklearn.impute import SimpleImputer

Sim = SimpleImputer(missing_values = np.nan, strategy = 'mean')
Sim = SimpleImputer().fit(x[:, 1:3])
x[:, 1:3] = Sim.transform(x[:, 1:3])

#2 Codificar datos categoricos

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

Labelencoder_x = LabelEncoder()
x[:, 0] = Labelencoder_x.fit_transform(x[:,0])

ct = ColumnTransformer([("Country", OneHotEncoder(),[1])], remainder = 'passthrough')
x = ct.fit_transform(x.tolist())

