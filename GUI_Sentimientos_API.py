# Importamos las librerías necesarias

import streamlit as st
from transformers import pipeline

# Cargar el modelo una sola vez y guardarlo en caché, evitando que carguemos el modelo cada ve que se ejecute la transformacion

@st.cache_resource

# Usaremos el modelo: clapAI/mmBERT-small-multilingual-sentiment para analisis de sentimientos

def cargar_modelo():

    modelo = "clapAI/mmBERT-small-multilingual-sentiment"

    return pipeline(
        "text-classification",
        model=modelo,
        device=-1
    )

sentimientos = cargar_modelo()

# Interfaz gráfica

# Titulos y subtitulos

st.title("ANÁLISIS DE SENTIMIENTOS")
st.write("Ingrese una frase para analizar el sentimiento:")
st.caption("Ejemplo: Estoy muy feliz con mi compra")

# Cuadro para ingresar el texto

texto = st.text_area("Texto", key="texto", height=120, label_visibility="collapsed")

# Boton limpiar

def limpiar(): st.session_state["texto"] = ""

# Posiciones de los botones

col1, col2 = st.columns([1, 3])

with col1: analizar = st.button("Analizar sentimiento")
with col2: st.button("Limpiar", on_click=limpiar)

# Analizar cuando se presiona el botón analizar

if analizar:

    if texto.strip():

        resultado = sentimientos(texto)[0]

        sentimiento = resultado["label"]
        confianza = resultado["score"] * 100

        clases = {
            "positive": "😁 Positivo",
            "neutral": "😐 Neutral",
            "negative": "😞 Negativo"
        }

        mostrar_sentimiento = clases.get(
            sentimiento,
            sentimiento
        )

        st.subheader("Resultado")

        st.write("Sentimiento:", mostrar_sentimiento)
        st.write("Confianza:", round(confianza, 2), "%")

    else:
        st.warning("Ingrese un texto para analizar.")
