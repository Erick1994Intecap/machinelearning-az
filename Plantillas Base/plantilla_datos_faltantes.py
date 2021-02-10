# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 23:19:28 2021

@author: Erick Ruiz Obregon
"""

##Plantilla de proprocesado - para datos faltantes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importacion de datos

data = pd.read_csv("Data.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#limpieza de datos o tratmiento de NaN

from sklearn.impute import SimpleImputer

#la estrategia tomara la media de los valores para poderlos reemplazar
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
