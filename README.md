# 🌽 Diagnóstico Fitosanitario de Maíz con IA y Grad-CAM

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)

> **ESPOL — FIEC · Inteligencia Artificial (Grupo #1)**  
> **Autores:** Ariana Palacios · Carlos Flores · Julio Guerrero  

Aplicación interactiva y modelo de Deep Learning para el diagnóstico temprano de enfermedades foliares en cultivos de maíz (*Zea mays*). Utiliza **Transfer Learning con MobileNetV2** y mapas de calor **Grad-CAM** para justificar visualmente cada predicción.

---

## 🚀 Resumen del Modelo y Resultados

* **Clases Diagnosticadas (5):** `Healthy` (Sana), `Common Rust` (Roya), `Asphalt Stain` (Mancha de Asfalto), `Bipolaris` y `Stenocarpella`.
* **Dataset:** 1,079 imágenes reales (partición estratificada 70% train / 15% val / 15% test).
* **Desempeño en Test (162 imágenes):** **88.27% Accuracy** y **87.95% F1-score macro**.
* **Impacto del Transfer Learning:** Supera en **+48.1 puntos de accuracy** a una CNN entrenada desde cero (*Baseline*: 40.12% vs *MobileNetV2*: 88.27%).

### Reporte de Clasificación (Test Set)

| Clase | Precisión | Recall | F1-Score | Muestras |
|---|:---:|:---:|:---:|:---:|
| **Healthy** | 92.1% | 100.0% | **95.9%** | 35 |
| **Stenocarpella** | 90.9% | 88.2% | **89.5%** | 34 |
| **Bipolaris** | 81.8% | 81.8% | **81.8%** | 33 |
| **Common Rust** | 80.6% | 96.7% | **87.9%** | 30 |
| **Asphalt Stain** | 100.0% | 73.3% | **84.6%** | 30 |
| **Global / Macro** | **89.1%** | **88.0%** | **88.0%** | **162** |

---

## 🧠 Arquitectura y Metodología

1. **Backbone Ligero:** MobileNetV2 preentrenado en ImageNet (~2.3M parámetros), ideal para despliegue rápido y baja memoria.
2. **Entrenamiento en Dos Fases:**
   - **Fase 1 (Extracción):** Backbone 100% congelado; se entrena solo la cabeza clasificadora (10 épocas, lr = $10^{-4}$) para estabilizar pesos.
   - **Fase 2 (Ajuste Fino):** Se descongelan quirúrgicamente los últimos 3 bloques (`block_14`, `block_15`, `block_16` y `Conv_1`) con lr = $10^{-5}$ y parada temprana.
   - **BatchNorm Congelada:** Se mantiene en modo inferencia para evitar ruido estadístico con lotes pequeños.
3. **Explicabilidad con Grad-CAM:** Calcula la activación de la última capa convolucional (`out_relu`) para generar un mapa de calor que resalta las lesiones foliares que sustentan la decisión.
4. **Preprocesamiento Embebido:** El reescalado $[-1, 1]$ viaja dentro del modelo (`.keras`), garantizando paridad exacta entre entrenamiento y la app.

---

## ⚡ Instalación Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/julioguerrero131/proyecto-IA-ESPOL.git
cd proyecto-IA-ESPOL

# 2. Crear y activar entorno virtual
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 🖥️ Uso de la Aplicación (Streamlit)

Inicia la interfaz gráfica ejecutando:

```bash
streamlit run main.py
```

Se abrirá en tu navegador `http://localhost:8501`.

### Características de la Interfaz:
- **Diagnóstico y Confianza:** Muestra la clase predicha y el porcentaje de certeza.
- **Alerta de Baja Certeza:** Si la confianza es menor al **60%**, sugiere repetir la captura a 15–20 cm sin sombras excesivas.
- **Panel Grad-CAM (3 columnas):**
  1. Captura original redimensionada ($224 \times 224$).
  2. Mapa de activación (calor).
  3. Superposición explicativa de la lesión sobre la hoja.
- **Top-3 de Probabilidades:** Desglose porcentual con barras de progreso.

### Probar con Imágenes de Ejemplo:
El proyecto incluye la carpeta [`examples/`](examples) con fotos listas para probar:
* `examples/healthy/`
* `examples/common-rust/`
* `examples/asphalt-stain/`
* `examples/bipolaris/`
* `examples/stenocarpella/`

---

## 📁 Estructura del Repositorio

```text
proyecto-IA-ESPOL/
├── README.md                      # Documentación general del proyecto
├── main.py                        # Aplicación web en Streamlit
├── requirements.txt               # Dependencias del proyecto
├── models/
│   └── best_maize_model_v5.keras  # Modelo final entrenado
├── notebook/
│   ├── README.md                  # Resumen técnico del entrenamiento
│   └── ProyectoIA_v5.ipynb        # Cuaderno de experimentación y entrenamiento
└── examples/                      # Imágenes de prueba clasificadas por patología
```

> Para más detalles sobre el pipeline de datos, experimentación y reproducibilidad del modelo, revisa el [README del Notebook](notebook/README.md).