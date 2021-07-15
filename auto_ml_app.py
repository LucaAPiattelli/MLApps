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

def main():
    st.title("Auto ML App")
    data = st.file_uploader("Carga tu Dataset", type=["csv","txt"])
    df = pd.read_csv(data)

    activites = "EDA", "Visualización", "Creación de modelo"
    choice = st.sidebar.selectbox("Selecciona lo que quieras realizar", activites)

    

# Creación del componente EDA    
    if choice == "EDA":
        st.subheader("Análisis Exploratorio de Datos")
        
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

# Creación del componente Visualización    
    elif choice == "Visualización":
        st.subheader("Visualización de Datos")

    if st.checkbox("Correlación"):
        st.write(sns.heatmap(df.corr(), annot = True))
        st.pyplot()

    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Elegí el tipo de gráfico", ["area","barras", "líneas", "histograma", "boxplot", "kde"])
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
        
        elif type_of_plot == "barras":
            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)
        
        elif type_of_plot:
            cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()
            

# Creación del componente Modelo ML  
    elif choice == "Creación de modelo":
        st.subheader("Creación de Modelo de Machine Learning")
    
        X = df.iloc[:,0:-1]
        Y = df.iloc[:,-1]
        seed = 23

        models = []
        models.append(("LR", LogisticRegression()))
        models.append(("LDA", LinearDiscriminantAnalysis()))
        models.append(("KNN", KNeighborsClassifier()))
        
        model_names = []
        model_mean = []
        model_std = []
        all_models = []
        scoring = "accuracy"

        for name, model in models:
            kfold = model_selection.KFold(n_splits = 10, random_state = seed)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring = scoring)
            model_names.append(name)
            model_mean.append(cv_results.mean())
            model_std.append(cv_results.std())

            accuracy_results = {"model_name":name, "model_accuracy":cv_results.mean(),"standard_deviation":cv_results.std()}
            all_models.append(accuracy_results)
        
        if st.checkbox("Ver métricas del modelo"):
            st.dataframe(pd.DataFrame(zip(model_names, model_mean, model_std), columns = ["Model Name", "Model Accuracy", "Standard Deviation"]))

if __name__ == "__main__":
    main()
