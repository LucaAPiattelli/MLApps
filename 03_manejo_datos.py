import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("Análisis de clientes")

# Importar datos
clientes = pd.read_csv("https://raw.githubusercontent.com/cristiandarioortegayubro/UNI/main/clientes.csv")
st.write(clientes)

# Crear un selector de valores
st.header("Utilizando Streamlit para crear tu propio gráfico")
variable_seleccionada = st.selectbox("Qué variable quieres visualizar?", ["Edad","Salario"])
variable_comparacion = st.selectbox("Con qué otra variable quieres compararlo?", ["Trabajo", "Compra"])
titulo_gráfico = variable_seleccionada +  " por " + variable_comparacion

# Visualizar 
x = variable_comparacion
y = variable_seleccionada
clientes = clientes[[x,y]]
clientes = clientes.groupby(x).mean()
clientes["x"] = clientes.index
clientes["y"] = clientes[clientes.columns[0]]

fig,ax = plt.subplots()
ax = plt.bar(x = clientes.x, height= clientes.y)
plt.xlabel(variable_comparacion)
plt.ylabel(variable_seleccionada)
plt.title(titulo_gráfico)
st.pyplot(fig)