import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Comando: streamlit run main.py

# 1. Carga del Modelo
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('models/best_maize_model_v1.keras')
    # return tf.keras.models.load_model('models/best_maize_model_v2.keras')

model = load_keras_model()

clases = ['Healthy', 'Stenocarpella', 'Bipolaris', 'Common Rust', 'Asphalt Satin']

# 2. Interfaz
st.title("Diagnóstico Fitosanitario de Hojas de Maíz")
st.write("Sube una imagen de la hoja afectada capturada a 15-20 cm de distancia.")

archivo_subido = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

if archivo_subido is not None:
    # Convertimos a RGB visualmente por si suben un PNG con transparencia (RGBA)
    img_visual = Image.open(archivo_subido).convert('RGB')
    st.image(img_visual, caption='Imagen a analizar', width=300)
    
    st.write("Analizando...")
    
    # 3. Preprocesamiento EXACTO al utilizado en el entrenamiento (TensorFlow nativo)
    # Extraemos los bytes puros de la imagen subida en la memoria
    img_bytes = archivo_subido.getvalue()
    
    # Decodificamos y redimensionamos forzando 3 canales (RGB)
    img_tensor = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img_resized = tf.image.resize(img_tensor, [224, 224])
    
    # Expandimos las dimensiones para crear el lote: (1, 224, 224, 3)
    img_batch = tf.expand_dims(img_resized, axis=0)
    
    # 4. Inferencia
    predicciones = model.predict(img_batch, verbose=0)[0]
    indice_clase = np.argmax(predicciones)
    clase_predicha = clases[indice_clase]
    confianza = predicciones[indice_clase] * 100
    
    # 5. Mostrar resultados
    st.success(f"Diagnóstico: **{clase_predicha}**")
    st.info(f"Nivel de confianza: {confianza:.2f}%")