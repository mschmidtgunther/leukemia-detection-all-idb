"""
================================================================================
PROYECTO: Detección y Clasificación Asistida de Leucemia Linfoblástica Aguda
ARCHIVO: main.py (Orquestador Central del Pipeline)
================================================================================
"""

import os
from pathlib import Path
import pandas as pd

# IMPORTACIÓN DE TUS MÓDULOS PROFESIONALES DESDE LA CARPETA SRC
from src.preprocessing import obtener_rutas_imagenes
from src.features import extract_dataset_features
from src.classification import clasificar_celula_v4  # Tu función experta V4 destilada
from src.evaluation import evaluar_rendimiento_sistema


def main():
    print("=" * 80)
    print("INICIANDO PIPELINE AUTOMATIZADO DE ANÁLISIS HEMATOLÓGICO (V6 DESTILADO)")
    print("=" * 80)
    
    # 1. Configuración de rutas reales del dataset
    # Ajusta esta ruta a donde tengas tus carpetas Benign, Early, Pre, Pro realmente
    RAW_PATH = Path("./data/raw/Original") 
    CLASES = ['Benign', 'Early', 'Pre', 'Pro']
    
    if not RAW_PATH.exists():
        # Si no existe esa ruta larga, probamos con una carpeta llamada 'dataset' en la raíz
        RAW_PATH = Path("./dataset")
        
    print(f"Buscando imágenes en la ruta: {RAW_PATH.resolve()}")
    
    # 2. Carga del Dataset usando TU función de preprocessing.py
    try:
        dataset, todas_las_rutas = obtener_rutas_imagenes(RAW_PATH, CLASES)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el dataset: {e}")
        return
        
    if len(todas_las_rutas) == 0:
        print("[ERROR] No se encontraron imágenes válidas. Verifica la ubicación de tus carpetas.")
        return

    # 3. Procesamiento y Extracción de Características (Preprocesamiento + Segmentación + Medición)
    # Corre tu bucle optimizado y genera el DataFrame
    df_robusto = extract_dataset_features(dataset)
    
    if df_robusto.empty:
        print("[ERROR] El DataFrame de características quedó vacío.")
        return
        
    print(f"-> Extracción finalizada. Matriz de datos generada: {df_robusto.shape[0]} células analizadas.")

    # 4. Inferencia del Sistema Experto (Tus reglas condicionales del Árbol de Decisión)
    print("-> Pasando características por el Motor de Inferencia V4 (If/Elif/Else)...")
    df_robusto['Prediccion'] = df_robusto.apply(clasificar_celula_v4, axis=1)

    # 5. Evaluación Científica y Mapeo de Resultados
    print("-> Generando reporte estadístico y matriz de confusión...")
    
    y_real = df_robusto["Clase"]
    y_pred = df_robusto["Prediccion"]
    
    # Llama a tu función de evaluation.py (Muestra gráfico y genera la tabla de Pandas)
    tabla_reporte = evaluar_rendimiento_sistema(
        y_real, 
        y_pred, 
        titulo_grafico="Matriz de Confusión: Sistema Experto V4 (Destilado por Datos)"
    )
    
    # Imprime la tabla final estilizada en la consola
    print("\n" + "=" * 60)
    print("TABLA DE RENDIMIENTO CIENTÍFICO FINAL")
    print("=" * 60)
    print(tabla_reporte)
    print("=" * 60)
    print("\n¡Pipeline ejecutado con éxito! El sistema está listo para la entrega.")


if __name__ == "__main__":
    main()