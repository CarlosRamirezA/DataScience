#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:45:30 2020

@author: carlos
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values


#MEtodo del codo para escoger la cantidad de cluster a usar

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Metodo del Codo")
plt.xlabel("Numro de Clusters")
plt.ylabel("WCSS(k)")
plt.show()

#Aplicar el metodo de k-means para segmentar el dataset

kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
    
#Visualizacion de clusters 

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 80, c = "red", label ="Normal")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 80, c = "blue", label ="Gastadores")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 80, c = "green", label ="Ricos")
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 80, c = "cyan", label ="Conservadores")
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 80, c = "magenta", label ="Tacaños")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 280, c = "yellow", label = "Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingreso anuales (en miles de $)")
plt.ylabel("Puntuacion de Gastos (1-100)")
plt.legend()
plt.show()