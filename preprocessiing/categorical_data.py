#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 18:32:29 2020

@author: carlos
"""
#Plantilla de Pre procesado - categoricos

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 
dataset = pd.read_excel('Book.xlsx')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values 

#2 Codificar datos categoricos y dummies

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

Labelencoder_x = LabelEncoder()
x[:, 0] = Labelencoder_x.fit_transform(x[:,0])

ct = ColumnTransformer([("Country", OneHotEncoder(),[0])], remainder = 'passthrough')
x = ct.fit_transform(x)

Labelencoder_y = LabelEncoder()
y = Labelencoder_y.fit_transform(y)