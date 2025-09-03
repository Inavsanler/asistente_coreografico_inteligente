#  Asistente Coreográfico Inteligente

Un asistente basado en **Computer Vision** y **Machine Learning** que analiza vídeos de danza y ofrece **sugerencias coreográficas** (alineación, extensión, ritmo, postura, expresión, etc.).

## ✨ Características

- 📥 **Entrada de datos**:  
  - CSV con características preprocesadas.  
  - Vídeo (`.mp4`, `.avi`, `.mov`) → extracción automática de **pose** con YOLOv8-Pose o MediaPipe.  

- 🧩 **Extracción de características**:  
  - Ventaneo configurable (tamaño y salto en segundos).  
  - Features básicas: velocidad, simetría, rango articular, confianza, missing rate.  

- 🤖 **Modelo entrenado**:  
  - Carga desde `artifacts/` (`joblib`, `labels.json`, `thresholds.json`).  
  - Soporta **umbral global** y **umbral por etiqueta** (`thresholds.json`).  

- 💡 **Sugerencias coreográficas**:  
  - Mapeo etiqueta → consejo personalizado.  
  - Configurable desde la barra lateral o vía JSON.  

- ⬇️ **Exportación**:  
  - Descargar CSV de features extraídas.  
  - Descargar resultados de inferencia.  

## 🗂️ Estructura del repo

.
├── app.py # Interfaz principal en Streamlit
├── src/
│ └── model.py # Lógica de carga y predicción
├── artifacts/ # Artefactos del modelo entrenado
│ ├── imputer.joblib
│ ├── scaler.joblib
│ ├── model_ovr_logreg.joblib
│ ├── labels.json
│ └── thresholds.json
├── requirements.txt # Dependencias de Python
├── packages.txt # Dependencias del sistema (ffmpeg, libgl1)
└── README.md # Este archivo
