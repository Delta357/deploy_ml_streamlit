import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Carrega o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
classes = iris.target_names

# Cria o modelo de classificação
model = RandomForestClassifier()
model.fit(X, y)

# Configuração do aplicativo Streamlit
def main():
    st.title("Aplicativo de Classificação de Flores Iris")
    st.sidebar.title("Parâmetros")
    
    # Entrada de dados do usuário
    sepal_length = st.sidebar.slider("Comprimento da Sépala", float(X[:, 0].min()), float(X[:, 0].max()), 5.4)
    sepal_width = st.sidebar.slider("Largura da Sépala", float(X[:, 1].min()), float(X[:, 1].max()), 3.0)
    petal_length = st.sidebar.slider("Comprimento da Pétala", float(X[:, 2].min()), float(X[:, 2].max()), 1.0)
    petal_width = st.sidebar.slider("Largura da Pétala", float(X[:, 3].min()), float(X[:, 3].max()), 0.2)
    
    # Realiza a previsão
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(data)
    predicted_class = classes[prediction[0]]
    
    # Exibe o resultado
    st.write("Valores inseridos pelo usuário:")
    input_data = pd.DataFrame(data, columns=["Comprimento da Sépala", "Largura da Sépala", "Comprimento da Pétala", "Largura da Pétala"])
    st.write(input_data)
    
    st.write("Classe prevista:")
    st.write(predicted_class)

if __name__ == "__main__":
    main()
