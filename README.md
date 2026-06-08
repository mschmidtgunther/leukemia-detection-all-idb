# Detección y Clasificación Asistida de Leucemia Linfoblástica Aguda (LLA)

Este proyecto implementa un pipeline automatizado de procesamiento de imágenes biomédicas para segmentar y extraer características morfológicas y de textura de glóbulos blancos a partir de la base de datos pública ALL-IDB.

## Base de datos

Para el desarrollo del proyecto se utilizó **ALL-IDB** (Acute Lymphoblastic Leukemia Image Database):

>  Los datos **no se suben al repo**. Solicitá acceso en [ALL-IDB](https://homes.di.unimi.it/scotti/all/) y colocalos en `data/raw/`.

## Interfaz

La interfaz desarrollada permite el analisis de imágenes individuales, múltiples imágenes y una evaluación del proyecto con una base de datos

- Clasificación individual: Sin requisitos
- Clasificación Grupal: Requiere que las imágenes esten en formato .jpg o .png y que se encuentren en:
`.data/clasificacion`

- Evaluación con Base de datos: Requiere que la base de datos se encuentre en:
`data/raw/Original`
- `/Benign`: Para frontis con células sanas
- `/Early`: Para frontis con Linfoblastos en etapa temprana
- `/Pre`: Para frontis con Linfoblastos pre-B o pre-T
- `/Pro`: Para frontis con Prolinfoblastos


## Estructura del Proyecto
* `data/`: Carpeta local para almacenar las imágenes (no se sube al repositorio).
* `notebooks/`: Espacio de experimentación en Jupyter.
* `src/`: Código fuente modularizado del pipeline (preprocesamiento, segmentación, características y evaluación).

## 🚀 Instalación
Para instalar las librerías necesarias, ejecutá en tu terminal:
```bash
pip install -r requirements.txt



