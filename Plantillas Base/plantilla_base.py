# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 12:04:21 2021

@author: Erick Ruiz Obregon
"""

##Plantilla de proprocesado
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importacion de datos

data = pd.read_csv("Data.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values



#dividir el dataset en conjunto de entrenamiento y testing
#0.2 quiere decir que se usara el 20% de los datos para pruebas
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#no siempre se van a escalar los datos
#esto se debe haceer cuando hay mucha distancia entre las variables
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
'''










