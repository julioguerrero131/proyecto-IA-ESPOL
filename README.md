# Proyecto IA ESPOL: Diagnóstico Fitosanitario de Hojas de Maíz

Este proyecto implementa una aplicación interactiva utilizando Streamlit que permite subir imágenes de hojas de maíz y utilizar un modelo de Inteligencia Artificial para diagnosticar posibles enfermedades (Stenocarpella, Bipolaris, Roya Común/Common Rust, Mancha de Asfalto/Asphalt Stain) o identificar si está sana (Healthy). Adicionalmente, la herramienta emplea la técnica Grad-CAM para resaltar en un mapa de calor la región de la imagen en la que el modelo se ha basado para realizar su diagnóstico.

## Requisitos e Instalación

Para instalar las dependencias necesarias del proyecto, ejecuta el siguiente comando en tu terminal (se recomienda el uso de un entorno virtual):

```bash
pip install -r requirements.txt
```

## ¿Cómo iniciar la interfaz?

Una vez que tengas todas las dependencias instaladas, puedes iniciar la interfaz gráfica desde la línea de comandos ejecutando:

```bash
streamlit run main.py
```

Automáticamente se abrirá una pestaña en tu navegador web por defecto mostrando la aplicación.

## ¿Cómo usar las imágenes de prueba (Carpeta `examples`)?

El repositorio incluye una carpeta llamada `examples` que contiene imágenes reales para que puedas probar el funcionamiento del modelo sin tener que buscar imágenes propias. Las imágenes están clasificadas en las siguientes subcarpetas:

- `asphalt-stain` (Mancha de Asfalto)
- `bipolaris`
- `common-rust` (Roya Común)
- `healthy` (Sano)
- `stenocarpella`

**Pasos para probar el modelo con estos ejemplos:**
1. Inicia la interfaz siguiendo las instrucciones anteriores.
2. En la página web de la aplicación, haz clic en el botón de **"Elige una imagen..."** o arrastra un archivo hacia la zona indicada.
3. Navega dentro de tu explorador de archivos hasta la ruta del proyecto y abre la carpeta `examples`.
4. Ingresa a la subcarpeta de la enfermedad que desees probar y selecciona una de las imágenes.
5. El modelo analizará la hoja y mostrará:
   - La predicción de la enfermedad.
   - El porcentaje de confianza de la predicción.
   - El mapa de calor (Grad-CAM) superpuesto sobre la imagen original, que indica qué zonas visuales influyeron más en la decisión del modelo.