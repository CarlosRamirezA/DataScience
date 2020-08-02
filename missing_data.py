#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 18:32:49 2020

@author: carlos
"""
#Plantilla de Pre procesado - tratamientos nan

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
