# 📓 Cuaderno de Entrenamiento: `ProyectoIA_v5.ipynb`

Este documento resume los aspectos técnicos, decisiones de diseño y resultados experimentales del cuaderno de entrenamiento.

---

## ⚙️ Configuración y Datos

* **Entorno:** Python 3.10+, TensorFlow 2.20+, GPU T4 (Google Colab).
* **Reproducibilidad:** Semilla global `SEED = 42`.
* **Dataset (1,079 imágenes):** Partición estratificada 70% entrenamiento (755), 15% validación (162) y 15% prueba (162).
  * `healthy`: 232 | `stenocarpella`: 227 | `bipolaris`: 220 | `common_rust`: 200 | `asphalt_stain`: 200
* **Balanceo de Clases:** Inyección de `sample_weight` en el generador `tf.data`, calculado únicamente sobre el conjunto de entrenamiento para evitar fuga de información.
* **Aumentación de Datos:** Volteo horizontal, rotación ($\pm 20^\circ$), zoom ($0.8 - 1.2$), brillo y contraste.

---

## 🏗️ Arquitectura y Entrenamiento en 2 Fases

```
Entrada (224x224x3) ➔ Rescaling [-1, 1] ➔ MobileNetV2 (Backbone) ➔ GAP2D ➔ Dropout(0.3) ➔ Dense(128, ReLU) ➔ Dropout(0.2) ➔ Softmax(5)
```

| Fase | Capas Activas | Optimizador / Tasa | Épocas | Objetivo |
|---|---|:---:|:---:|---|
| **Fase 1: Extracción** | Solo cabeza densa (164k params). Backbone 100% congelado. | Adam, $\text{lr} = 10^{-4}$ | 10 | Estabilizar la cabeza clasificadora sin corromper pesos de ImageNet. |
| **Fase 2: Fine-Tuning** | Cabeza + últimos 3 bloques de MobileNetV2 (`block_14`, `block_15`, `block_16`, `Conv_1`). | Adam, $\text{lr} = 10^{-5}$ | 30 (parada temprana en época 15) | Readaptar características profundas a la morfología de las lesiones fúngicas. |

> **Nota técnica:** Las capas `BatchNormalization` se mantuvieron congeladas (`trainable = False`) en ambas fases para evitar inestabilidad estadística con lotes pequeños ($N=32$).

---

## 🔬 Evidencias Experimentales

### 1. Justificación de Transfer Learning vs. Desde Cero
Para demostrar numéricamente el valor de MobileNetV2 frente a un entrenamiento desde cero, se entrenó una CNN equivalente como control:

* **Baseline CNN (desde cero):** 40.12% Accuracy / 34.79% F1-score macro.
* **MobileNetV2 (Transfer Learning):** **88.27% Accuracy / 87.95% F1-score macro**.
* **Ganancia:** **+48.1 puntos porcentuales** en exactitud.

### 2. Selección del Número de Bloques a Descongelar
Se exploró el nivel óptimo de descongelamiento seleccionando el mejor candidato **exclusivamente por exactitud de validación** (sin tocar el conjunto de prueba durante la búsqueda):

| Configuración | Bloques Descongelados | val_accuracy | val_loss | Resultado |
|---|:---:|:---:|:---:|:---:|
| `ultimo_1_bloque` | block_16 + Conv_1 | 85.80% | 0.3706 | Sub-adaptado |
| `ultimos_5_bloques` | block_12 al 16 + Conv_1 | 90.12% | 0.2824 | Sobreparametrizado |
| **`ultimos_3_bloques`** | **block_14 al 16 + Conv_1** | **90.74%** | **0.3134** | **Ganador Seleccionado** |

---

## 🔍 Explicabilidad con Grad-CAM

Para auditar visualmente las decisiones del modelo y evitar el efecto "caja negra":
1. Se calculan los gradientes de la clase predicha con respecto a la última capa convolucional (`out_relu`).
2. Se realiza un promedio espacial (*Global Average Pooling*) para obtener la ponderación de importancia de cada canal.
3. Se aplica $\text{ReLU}$ y normalización $[0, 1]$ para resaltar solo la evidencia favorable.
4. El mapa de calor resultante se superpone sobre la hoja, confirmando que la red focaliza pústulas y manchas necróticas y no el fondo.

---

## 💾 Exportación

El modelo final se guarda en formato nativo Keras:
* Archivo generado: `best_maize_model_v5.keras`
* Ubicación para inferencia: [`models/best_maize_model_v5.keras`](../models/best_maize_model_v5.keras)
* Compatible directamente con la aplicación en [`main.py`](../main.py).
