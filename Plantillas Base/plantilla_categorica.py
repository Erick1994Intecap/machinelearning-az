# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 23:17:54 2021

@author: Erick Ruiz Obregon
"""

    ##Plantilla de proprocesado - datos categoricos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importacion de datos

data = pd.read_csv("Data.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#Codificacion de datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
x = onehotencoder.fit_transform(x)

# Evitar la trampa de las variables ficticias
x = x[:, 1:]


#codificacion de y

codificador_y = LabelEncoder()
y = codificador_y.fit_transform(y)
