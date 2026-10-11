# 📓 Documentación Técnica: Cuaderno de Entrenamiento (`ProyectoIA_v5.ipynb`)

Este documento complementa el [README principal](../README.md) y detalla la fundamentación teórica, las decisiones de implementación y la evidencia experimental contenida en el cuaderno interactivo de Jupyter/Colab: **`notebook/ProyectoIA_v5.ipynb`**.

---

## 📑 Contenidos

1. [Entorno de Ejecución y Dependencias](#1-entorno-de-ejecución-y-dependencias)
2. [Auditoría del Dataset y Partición Estratificada](#2-auditoría-del-dataset-y-partición-estratificada)
3. [Generadores `tf.data` y Balanceo de Clases](#3-generadores-tfdata-y-balanceo-de-clases)
4. [Aumentación de Datos y Preprocesamiento Nativo](#4-aumentación-de-datos-y-preprocesamiento-nativo)
5. [Arquitectura del Modelo y Justificaciones Académicas](#5-arquitectura-del-modelo-y-justificaciones-académicas)
6. [Entrenamiento en Dos Fases](#6-entrenamiento-en-dos-fases)
7. [Experimento Control: Baseline sin Transferencia](#7-experimento-control-baseline-sin-transferencia)
8. [Búsqueda de Bloques de Fine-Tuning (Sin Fuga de Datos)](#8-búsqueda-de-bloques-de-fine-tuning-sin-fuga-de-datos)
9. [Formulación e Implementación de Grad-CAM](#9-formulación-e-implementación-de-grad-cam)
10. [Métricas de Desempeño y Evaluación Final](#10-métricas-de-desempeño-y-evaluación-final)
11. [Inferencia Autónoma y Casos de Uso](#11-inferencia-autónoma-y-casos-de-uso)

---

## 1. Entorno de Ejecución y Dependencias

El cuaderno está optimizado para ejecutarse en entornos GPU (Google Colab con GPU NVIDIA T4 o entornos locales con aceleración CUDA):

- **TensorFlow:** `2.20.0`
- **Keras:** `3.x`
- **NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn**
- **Semilla Global de Reproducibilidad:** `SEED = 42` (establecida mediante `keras.utils.set_random_seed(SEED)` para fijar la partición, inicialización de tensores y aumentaciones estocásticas).

---

## 2. Auditoría del Dataset y Partición Estratificada

El conjunto de datos base corresponde a **CornLeafDiseaseCollection**. La auditoría automatizada del sistema detectó la distribución muestral real frente a la descripción preliminar del documento:

| Clase | Nombre Científico / Diagnóstico | Imágenes Reales | Muestras Declaradas en Doc. |
|---|---|:---:|:---:|
| `healthy` | Hoja Sana | 232 | 200 |
| `stenocarpella` | *Stenocarpella macrospora* | 227 | 200 |
| `bipolaris` | *Bipolaris maydis* | 220 | 200 |
| `common_rust` | *Puccinia sorghi* / Southern Rust | 200 | 200 |
| `asphalt_stain` | *Phyllachora maydis* (Mancha de Asfalto) | 200 | 200 |
| **Total** | | **1,079** | **1,000** |

### Partición Estratificada (70% / 15% / 15%)
Se aplica `train_test_split` estratificado para preservar la distribución de clases en todas las etapas:
- **Entrenamiento (70%):** 755 imágenes
- **Validación (15%):** 162 imágenes
- **Prueba (15%):** 162 imágenes

---

## 3. Generadores `tf.data` y Balanceo de Clases

Para garantizar un flujo de datos asíncrono y de alto rendimiento hacia la GPU, se construyó una canalización basada en la API `tf.data.Dataset`:

```python
AUTOTUNE = tf.data.AUTOTUNE

def build_dataset(df_subset, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((df_subset['filepath'].values, df_subset['label'].values))
    if is_training:
        ds = ds.shuffle(len(df_subset), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_cargar, num_parallel_calls=AUTOTUNE)
    if is_training:
        ds = ds.map(_con_peso, num_parallel_calls=AUTOTUNE).repeat()
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
```

### Compensación de Desbalance Muestral
Dado que Keras no admite `class_weight` directo cuando las etiquetas están codificadas en one-hot con `CategoricalCrossentropy`, se calculan pesos balanceados **únicamente sobre el subconjunto de entrenamiento**:

$$w_c = \frac{N_{\text{train}}}{K \cdot N_{c,\text{train}}}$$

Donde $N_{\text{train}} = 755$, $K = 5$. Los pesos resultantes asignados a cada clase son:
- `healthy`: 0.932
- `stenocarpella`: 0.950
- `bipolaris`: 0.981
- `common_rust`: 1.079
- `asphalt_stain`: 1.079

Estos pesos se suministran al entrenamiento a través del tensor `sample_weight` dentro del mapa de `tf.data`, penalizando más los errores cometidos en clases minoritarias.

---

## 4. Aumentación de Datos y Preprocesamiento Nativo

Para evitar que el modelo dependa de código de normalización externo en la aplicación web, el reescalado y las transformaciones espaciales residen **dentro del grafo del modelo**:

1. **Transformaciones Geométricas:**
   - `RandomFlip('horizontal')`: Reflejo especular horizontal.
   - `RandomRotation(factor=20/360, fill_mode='reflect')`: Rotación aleatoria en el rango $[-20^\circ, +20^\circ]$.
   - `RandomZoom((-0.2, 0.2), (-0.2, 0.2), fill_mode='reflect')`: Factor de escala entre 0.8 y 1.2.
2. **Transformaciones Fotométricas:**
   - `RandomBrightness(0.2, value_range=(0, 255))`: Simula variaciones de luminosidad en tomas de campo.
   - `RandomContrast(0.2)`: Simula cambios de contraste producidos por cámaras de teléfonos móviles.
3. **Capa `Rescaling(1.0 / 127.5, offset=-1.0)`:** Transforma el rango $[0, 255]$ a $[-1, 1]$, que es el formato exigido por MobileNetV2.

---

## 5. Arquitectura del Modelo y Justificaciones Académicas

La estructura de la red sintetiza la respuesta a los requerimientos formales del proyecto:

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 entrada_hoja (InputLayer)   (None, 224, 224, 3)       0         
 data_augmentation (Seq)     (None, 224, 224, 3)       0         
 preprocesamiento (Rescaling (None, 224, 224, 3)       0         
 mobilenetv2_1.00_224 (Func) (None, 7, 7, 1280)        2,257,984 
 reduccion_espacial (GAP2D)  (None, 1280)              0         
 regularizacion_1 (Dropout)  (None, 1280)              0         
 densa_relu (Dense)          (None, 128)               163,968   
 regularizacion_2 (Dropout)  (None, 128)               0         
 softmax_classifier (Dense)  (None, 5)                 645       
=================================================================
Total params: 2,422,597 (9.24 MB)
```

- **Cabeza Clasificadora Compacta:** Consta de 164,613 parámetros entrenables. Los dos niveles de Dropout ($p=0.3$ y $p=0.2$) evitan la coadaptación de unidades densas ante el tamaño moderado del dataset.
- **Backbone Ligero:** MobileNetV2 aporta 2.25 millones de parámetros preentrenados, evitando la saturación de memoria de GPUs modestas.

---

## 6. Entrenamiento en Dos Fases

### Fase 1: Extracción de Características
- **Condición:** Backbone MobileNetV2 completamente congelado (`base_model.trainable = False`).
- **Hiperparámetros:** Optimizador Adam, $\text{Learning Rate} = 10^{-4}$, 10 épocas, `batch_size = 32`.
- **Evolución:** La exactitud de validación pasa de **49.38%** en la época 1 a **82.10%** en la época 10 (`val_loss = 0.5364`), permitiendo que la capa densa y el clasificador softmax aprendan hiperplanos separadores estables antes de perturbar los filtros convolucionales.

### Fase 2: Ajuste Fino (Fine-Tuning)
- **Condición:** Descongelamiento de las capas profundas seleccionadas por nombre (`block_14`, `block_15`, `block_16` y `Conv_1`).
- **Batch Normalization Congelada:** Todas las capas `layers.BatchNormalization` se mantienen con `trainable = False` para no sobreescribir las medias y varianzas estimadas sobre ImageNet con mini-lotes ruidosos de 32 imágenes.
- **Hiperparámetros:** Tasa de aprendizaje reducida $\eta = 10^{-5}$, hasta 30 épocas, *Early Stopping* con paciencia 5 monitoreando `val_loss`.
- **Resultado:** Convergencia óptima alcanzada en la **época 15** con **`val_loss = 0.2745`** y **`val_accuracy = 90.74%`**.

---

## 7. Experimento Control: Baseline sin Transferencia

Para demostrar analíticamente la necesidad de utilizar *Transfer Learning*, en la sección §13 se entrenó una red convolucional compacta desde cero:
- **Estructura:** 4 bloques `Conv2D(3x3) + MaxPooling2D` con 32, 64, 128 y 128 filtros, seguidos de `GlobalAveragePooling2D`, `Dense(128, relu)`, `Dropout(0.3)` y `Dense(5, softmax)`.
- **Resultados en Test:**
  - **Exactitud (Accuracy):** **40.12%**
  - **F1-Score Macro:** **34.79%**
- **Conclusión:** El modelo sin transferencia colapsa en la discriminación de patologías foliares complejas debido a la falta de regularización y volumen de datos, confirmando una **ventaja empírica de +48.15 puntos de exactitud** a favor de MobileNetV2.

---

## 8. Búsqueda de Bloques de Fine-Tuning (Sin Fuga de Datos)

En la sección §14 se ejecutó una búsqueda en cuadrícula del punto óptimo de descongelamiento:
1. `ultimo_1_bloque` (desde bloque 16: 7 capas liberadas)
2. `ultimos_3_bloques` (desde bloque 14: 19 capas liberadas)
3. `ultimos_5_bloques` (desde bloque 12: 31 capas liberadas)

> **Criterio Metodológico Riguroso:** La configuración ganadora se determinó **exclusivamente por la métrica de validación** (`val_accuracy`), corrigiendo la práctica errónea de seleccionar hiperparámetros mirando el conjunto de prueba (que generaría fuga de información).

La configuración **`ultimos_3_bloques`** obtuvo el mejor equilibrio entre capacidad de generalización y prevención de sobreajuste (`val_accuracy = 0.9074`, `val_loss = 0.3134`).

---

## 9. Formulación e Implementación de Grad-CAM

Grad-CAM localiza visualmente las áreas del tensor espacial que mayor contribución aportan a la activación de la neurona de salida correspondiente a la clase $c$:

```python
def gradcam_mapa(modelo, imagen_224, indice_clase=None):
    base = obtener_backbone(modelo)
    extractor = keras.Model(base.inputs, base.get_layer('out_relu').output)
    cabeza = [modelo.get_layer(n) for n in NOMBRES_CABEZA]

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
    pesos = tf.reduce_mean(grads, axis=(1, 2))  # Importancia por canal (GAP)
    mapa = tf.reduce_sum(mapas_conv * pesos[:, tf.newaxis, tf.newaxis, :], axis=-1)[0]
    mapa = tf.nn.relu(mapa)                    # Solo evidencia positiva
    mapa = mapa / (tf.reduce_max(mapa) + 1e-8)  # Normalización [0, 1]
    return mapa.numpy(), indice_clase, y[0].numpy()
```

### Validación Agronómica
Al superponer el mapa de calor sobre las capturas foliares de prueba, se comprueba que:
- En **Roya Común**, las altas activaciones rodean las pústulas alargadas dispersas en el haz.
- En **Mancha de Asfalto**, el calor converge sobre los microestromas circulares negruzcos.
- En **Hojas Sanas**, las activaciones se dispersan de manera uniforme y con baja intensidad sobre la venación foliar, sin concentrarse en focos lesionales.

---

## 10. Métricas de Desempeño y Evaluación Final

Evaluación honesta e imparcial sobre el conjunto de prueba (**162 imágenes independientes**, nunca antes vistas por el optimizador):

| Clase | Precisión | Sensibilidad (Recall) | F1-Score | Soporte |
|---|:---:|:---:|:---:|:---:|
| **Healthy** | 0.9211 | 1.0000 | 0.9589 | 35 |
| **Stenocarpella** | 0.9091 | 0.8824 | 0.8955 | 34 |
| **Bipolaris** | 0.8182 | 0.8182 | 0.8182 | 33 |
| **Common Rust** | 0.8056 | 0.9667 | 0.8788 | 30 |
| **Asphalt Stain** | 1.0000 | 0.7333 | 0.8462 | 30 |
| **Promedio Macro** | **0.8908** | **0.8801** | **0.8795** | 162 |
| **Promedio Ponderado** | **0.8908** | **0.8827** | **0.8812** | 162 |
| **Exactitud Global** | | | **88.27%** | 162 |

---

## 11. Inferencia Autónoma y Casos de Uso

La sección §16 define la función integral `diagnosticar(modelo, ruta_imagen, explicar=True)` que cubre los casos de uso formales del sistema:
- **CU-01 (Carga):** Decodificación protegida con manejo de excepciones para archivos corruptos o formatos no estándar.
- **CU-02 (Diagnóstico):** Cálculo de probabilidades y advertencia si la confianza máxima es $< 60\%$ (`UMBRAL_BAJA_CONFIANZA`).
- **CU-03 (Explicabilidad):** Generación y despliegue inmediato del mapa Grad-CAM para auditoría visual.

El artefacto final se exporta como `best_maize_model_v5.keras` y se aloja en el directorio `models/` para su consumo directo por `main.py`.
