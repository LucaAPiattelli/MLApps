import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pycaret.classification import * 


# url = https://raw.githubusercontent.com/cristiandarioortegayubro/UNI/main/clientes.csv


def main():
    st.title("Auto ML App")
    st.subheader("Carga de datos")
    data = st.text_input("Carga tu Dataset")
    df = pd.read_csv(data)

    activites = "EDA", "Visualización", "Creación de modelo"
    choice = st.sidebar.selectbox("Selecciona lo que quieras realizar", activites)

# EDA
    if choice == "EDA":
        st.header("Análisis Exploratorio de Datos")
        
        if st.checkbox("Mostrar Encabezado"):
            st.write(df.head())

        if st.checkbox("Mostrar Forma"):
            st.write(df.shape)

        if st.checkbox("Mostrar Columnas"):
            all_columns = df.columns.to_list()
            st.write(all_columns)
        
        if st.checkbox("Seleccionar columnas para mostrar"):
            selected_columns = st.multiselect("Selecciona columnas", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        if st.checkbox("Mostrar Resumen"):
            st.write(df.describe())

        if st.checkbox("Mostrar Recuento de Valores"):
            st.write(df.iloc[:,-1].value_counts())

# Visualización

    elif choice == "Visualización":
        st.header("Visualización de Datos")

        if st.checkbox("Correlación"):
            fig, ax = plt.subplots()
            ax = sns.heatmap(df.corr(), annot = True)
            plt.title("Correlación entre variables")
            st.pyplot(fig)

        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Elegí el tipo de gráfico", ["area","barras", "líneas", "histograma", "boxplot"])
        selected_columns_names = st.multiselect("Elegí las columnas a graficar", all_columns_names)

        if st.button("Generar gráfico"):
            st.success("Generando el gráfico de {} para {}".format(type_of_plot, selected_columns_names))

        if type_of_plot == "area":
            cust_data = df[selected_columns_names]
            st.area_chart(cust_data)

        elif type_of_plot == "barras":
            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)

        elif type_of_plot == "líneas":
            cust_data = df[selected_columns_names]
            st.line_chart(cust_data)
        
        elif type_of_plot == "histograma":
            cust_data = df[selected_columns_names]
            fig,ax = plt.subplots()
            ax = plt.hist(cust_data)
            st.pyplot(fig)

        elif type_of_plot == "boxplot":
            cust_data = df[selected_columns_names]
            fig,ax = plt.subplots()
            ax = plt.boxplot(cust_data)
            st.pyplot(fig)

# Machine Learning

    elif choice == "Creación de modelo":
        st.header("Creación de Modelo de Machine Learning")

        st.subheader("Selección de datos")
        all_columns_names = df.columns.tolist()
        selected_columns_df = st.multiselect("Elegí las variables para tu dataset", all_columns_names)
        df_modelo = df[selected_columns_df]
        st.write(df_modelo)
        selected_columns_y = st.multiselect("Elegí la variable objetivo", all_columns_names)
        y = selected_columns_y[0]
    
        #selected_columns_cat = st.multiselect("Elegí las variables categoricas", all_columns_names)
        #selected_columns_num = st.multiselect("Elegí las variables numéricas", all_columns_names)
        

        st.subheader("Creación del modelo")

        # crear modelo
        classification = setup(data = df_modelo, target = y, silent = True, preprocess = False) #categorical_features = selected_columns_cat, numeric_features = selected_columns_num)
        mejor_modelo = compare_models()
        st.write("El mejor modelo es:", mejor_modelo)

        # elegir el nombre del modelo a crear
        modelos = pd.DataFrame(models())
        modelos = modelos.reset_index()
        st.write(modelos)
        modelo_seleccionado = st.multiselect("Elegí el modelo que queres crear", modelos)
        modelo = create_model(modelo_seleccionado[0])

        prediccion = predict_model(modelo)
        st.write(prediccion)
        
        # Calcular métrica


if __name__ == "__main__":
    main()