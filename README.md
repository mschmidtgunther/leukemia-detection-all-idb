# Detección y Clasificación Asistida de Leucemia Linfoblástica Aguda (LLA)

Este proyecto implementa un pipeline automatizado de procesamiento de imágenes biomédicas para segmentar y extraer características morfológicas y de textura de glóbulos blancos a partir de la base de datos pública ALL-IDB.

## Base de datos

**ALL-IDB** (Acute Lymphoblastic Leukemia Image Database):
- Imágenes microscópicas de frotis sanguíneos teñidos
- Anotaciones manuales (Ground Truth) por expertos
- ALL-IDB1 (segmentación) y ALL-IDB2 (clasificación)

> ⚠️ Los datos **no se suben al repo**. Solicitá acceso en [ALL-IDB](https://homes.di.unimi.it/scotti/all/) y colocalos en `data/raw/`.


## 📁 Estructura del Proyecto
* `data/`: Carpeta local para almacenar las imágenes (no se sube al repositorio).
* `notebooks/`: Espacio de experimentación en Jupyter.
* `src/`: Código fuente modularizado del pipeline (preprocesamiento, segmentación, características y evaluación).

## 🚀 Instalación
Para instalar las librerías necesarias, ejecutá en tu terminal:
```bash
pip install -r requirements.txt
