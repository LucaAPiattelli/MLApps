import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Ilustración del teorema del límite central")
st.subheader("Una aplicación de los alumnos de la UNI")
st.write("Esta aplicación simula mil lanzamientos de una moneda dada la probabilidad de que sea cara ingresada a continuación y el promedio de estos lanzamientos para ilustrar el teorema del límite central")
porcentaje_caras = st.number_input(label = "Cuál es la probabilidad de que una moneda sea cara?", min_value = 0.0, max_value = 1.0, value = 0.5)
titulo_grafico = st.text_input(label = "Qué título querés colocarle al gráfico?")
binom_dist = np.random.binomial(1, porcentaje_caras, 1000)
lista_medias = []
for i in range(0,1000):
    lista_medias.append(np.random.choice(binom_dist, 100, replace = True).mean())
fig, ax = plt.subplots()
ax = plt.hist(lista_medias)
plt.title(titulo_grafico)
st.pyplot(fig)
st.write("Promedio de la distribución: ", round(np.mean(lista_medias), 3))