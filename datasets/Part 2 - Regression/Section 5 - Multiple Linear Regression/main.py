"""@author

: Erick
Ruiz
Obregon
"""

##Plantilla de proprocesado - datos categoricos
import matplotlib.pyplot as plt
import pandas as pd
from funciones import *

# importacion de datos

data = pd.read_csv("50_Startups.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Codificacion de datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder="passthrough")
x = onehotencoder.fit_transform(x)

# Evitar la trampa de las variables ficticias
x = x[:, 1:]

# construir un modelo optimo de regresion multiple con el modelo de eliminacion hacia atras
# se agreaga una fila de unos a la matrix de x, se deja en arr para que lo deje al comienzo
x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)

sl = 0.05  # variable de significacion(limite para p valor)

# X optimo, todas las filas, y todas lass columnas con la intecion de ir eliminando columnas
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination_RS(x_opt, y)

# dividir el dataset en conjunto de entrenamiento y testing
# 0.2 quiere decir que se usara el 20% de los datos para pruebas
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_Modeled, y, test_size=0.2, random_state=0)

# ajustar el moedelo de regresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression

regresion = LinearRegression()
regresion.fit(x_train, y_train)

# prediccion de resultados
y_pred = regresion.predict(x_test)
