import streamlit as st
from transformers import pipeline
import torch


# Modelo
modelo = "tabularisai/multilingual-sentiment-analysis"


# Cargar modelo
sentimientos = pipeline(
    "sentiment-analysis",
    model=modelo,
    device=0 if torch.cuda.is_available() else -1
)


# Título
st.title("Análisis de Sentimientos")

st.write("Ingrese una frase para analizar el sentimiento:")


# Casilla de texto
texto = st.text_area(
    "Texto:",
    placeholder="Ejemplo: Estoy muy feliz con el producto que compré."
)


# Botón
if st.button("Analizar sentimiento"):

    if texto:

        # Analizar sentimiento
        resultado = sentimientos(texto)[0]

        # Obtener resultado
        sentimiento = resultado["label"]
        confianza = resultado["score"] * 100

        # Mostrar resultado
        st.subheader("Resultado")

        st.write("Sentimiento:", sentimiento)
        st.write("Confianza:", round(confianza, 2), "%")

    else:
        st.warning("Ingrese un texto para analizar.")
