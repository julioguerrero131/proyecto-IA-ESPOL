import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

# Comando de ejecución: streamlit run main.py

st.set_page_config(page_title="Diagnóstico de Maíz + Grad-CAM", layout="wide")

# ==============================================================================
# 1. Configuración y Constantes del Modelo
# ==============================================================================
IMG_SIZE = (224, 224)
CAPA_CONV_FINAL = 'out_relu'
NOMBRES_CABEZA = [
    'reduccion_espacial', 'regularizacion_1', 'densa_relu',
    'regularizacion_2', 'softmax_classifier'
]
CLASES = ['Healthy', 'Stenocarpella', 'Bipolaris', 'Common Rust', 'Asphalt Stain']
UMBRAL_BAJA_CONFIANZA = 60.0  # Umbral de certeza mínima (60%)

# ==============================================================================
# 2. Funciones de Carga y Grad-CAM
# ==============================================================================
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('models/best_maize_model_v5.keras')

model = load_keras_model()

def obtener_backbone(modelo):
    """Localiza la MobileNetV2 anidada dentro del modelo cargado."""
    for capa in modelo.layers:
        if isinstance(capa, tf.keras.Model) and 'mobilenet' in capa.name.lower():
            return capa
    raise ValueError("No se encontró MobileNetV2 dentro del modelo.")

def generar_gradcam(modelo, imagen_224, indice_clase=None):
    """Calcula los gradientes y genera el mapa de activación Grad-CAM."""
    base = obtener_backbone(modelo)
    extractor = tf.keras.Model(base.inputs, base.get_layer(CAPA_CONV_FINAL).output)
    cabeza = [modelo.get_layer(n) for n in NOMBRES_CABEZA]

    # Preprocesamiento a escala [-1, 1] requerida por MobileNetV2
    x = tf.expand_dims(tf.cast(imagen_224, tf.float32), axis=0)
    x = x / 127.5 - 1.0

    with tf.GradientTape() as cinta:
        mapas_conv = extractor(x, training=False)
        cinta.watch(mapas_conv)
        y = mapas_conv
        for capa in cabeza:
            y = capa(y, training=False)
        if indice_clase is None:
            indice_clase = int(tf.argmax(y[0]))
        puntuacion = y[:, indice_clase]

    grads = cinta.gradient(puntuacion, mapas_conv)
    pesos = tf.reduce_mean(grads, axis=(1, 2))
    mapa = tf.reduce_sum(mapas_conv * pesos[:, tf.newaxis, tf.newaxis, :], axis=-1)[0]
    mapa = tf.nn.relu(mapa)
    mapa = mapa / (tf.reduce_max(mapa) + 1e-8)
    return mapa.numpy(), indice_clase, y[0].numpy()

def superponer_mapa(mapa, imagen_original_224, alpha=0.4):
    """Genera la imagen RGB del mapa de calor y la superposición con la original."""
    # Redimensionar mapa al tamaño de la imagen original
    calor = tf.image.resize(mapa[..., tf.newaxis], IMG_SIZE).numpy()[..., 0]
    
    # Obtener el mapa de color 'jet' de forma compatible con cualquier versión de Matplotlib
    try:
        colormap = mpl.colormaps['jet']
    except (AttributeError, KeyError):
        colormap = plt.get_cmap('jet')
        
    calor_coloreado = colormap(calor)[:, :, :3]
    calor_rgb = np.uint8(255 * calor_coloreado)
    
    img_orig = imagen_original_224.astype('uint8')
    # Fusión ponderada
    superposicion = (calor_rgb * alpha + img_orig * (1.0 - alpha)).astype('uint8')
    return calor_rgb, superposicion

# ==============================================================================
# 3. Interfaz de Usuario en Streamlit
# ==============================================================================
st.title("Diagnóstico Fitosanitario de Hojas de Maíz con Grad-CAM")
st.write("Sube una imagen de la hoja afectada capturada a 15-20 cm de distancia con buena iluminación.")

archivo_subido = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

if archivo_subido is not None:
    try:
        # Preprocesamiento TensorFlow idéntico al entrenamiento
        img_bytes = archivo_subido.getvalue()
        img_tensor = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img_resized = tf.image.resize(img_tensor, IMG_SIZE).numpy()

        with st.spinner("Procesando imagen y calculando explicabilidad visual (Grad-CAM)..."):
            # Generar Grad-CAM y obtener predicciones
            mapa_calor, idx_pred, probabilidades = generar_gradcam(model, img_resized)
            calor_rgb, img_superpuesta = superponer_mapa(mapa_calor, img_resized, alpha=0.45)

        clase_predicha = CLASES[idx_pred]
        confianza = float(probabilidades[idx_pred]) * 100.0

        # --- Diagnóstico Principal ---
        st.subheader("Resultados del Diagnóstico")
        st.success(f"Diagnóstico: **{clase_predicha}**")
        st.info(f"Nivel de confianza: **{confianza:.2f}%**")

        # Advertencia de baja certeza
        if confianza < UMBRAL_BAJA_CONFIANZA:
            st.warning(
                f"**Advertencia:** La certeza está por debajo del umbral recomendado ({UMBRAL_BAJA_CONFIANZA:.0f}%). "
                "Repita la captura a 15-20 cm de distancia, enfocada y sin sombras excesivas."
            )

        # --- Visualización Grad-CAM (3 Columnas) ---
        st.markdown("### 🔬 Explicabilidad Visual (Grad-CAM)")
        st.caption("El mapa de calor resalta las regiones del tejido foliar en las que la red neuronal basó su decisión.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_resized.astype('uint8'), caption="1. Captura original (224x224)", use_container_width=True)
        with col2:
            st.image(calor_rgb, caption="2. Mapa de activación (Calor)", use_container_width=True)
        with col3:
            st.image(img_superpuesta, caption="3. Superposición explicativa", use_container_width=True)

        # --- Distribución de Probabilidades Top-3 ---
        with st.expander("Ver distribución de probabilidades (Top-3)"):
            top3_indices = np.argsort(probabilidades)[::-1][:3]
            for i in top3_indices:
                st.write(f"- **{CLASES[i]}:** {probabilidades[i] * 100:.2f}%")
                st.progress(float(probabilidades[i]))

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")