"Extracción de características y clasificación para el caso de que son muchas imágenes sin etiquetas"

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Importamos tu pipeline de procesamiento existente
from src.preprocessing import preprocesar_imagen
from src.features import calcular_porcentaje_rosado_total
from src.classification import clasificar_celula_v4
from src.segmentation import (
    segmentar_otsu_vs_watershed, 
    filtrar_linfocitos, 
    expandir_mascara_celula, 
    segmentar_nucleo_agresivo, 
    aislar_citoplasma_real
)

def procesar_lote_ciego(ruta_carpeta):
    """
    Lee todas las imágenes en una carpeta, aplica el pipeline de procesamiento
    y devuelve un DataFrame con el diagnóstico de cada una.
    """
    ruta = Path(ruta_carpeta)
    if not ruta.exists():
        return pd.DataFrame() # Devuelve vacío si no existe la carpeta
    
    # Buscar todas las imágenes jpg y png
    imagenes = list(ruta.glob('*.jpg')) + list(ruta.glob('*.png'))
    if len(imagenes) == 0:
        return pd.DataFrame()

    resultados = []

    for p in imagenes:
        try:
            # 1. Pipeline de Procesamiento
            img_original, img_preprocesada = preprocesar_imagen(p)
            _, mask_ws = segmentar_otsu_vs_watershed(img_original, img_preprocesada, fg_factor=0.52)
            mask_base = filtrar_linfocitos(mask_ws, min_area=200, min_circularidad=0.75)
            mask_zona_busqueda = expandir_mascara_celula(mask_base, iteraciones=1)
            mask_nucleo_bruto = segmentar_nucleo_agresivo(img_preprocesada, ajuste_otsu=-10)
            mask_nucleo_final = cv2.bitwise_and(mask_nucleo_bruto, mask_zona_busqueda)
            mask_citoplasma = aislar_citoplasma_real(img_original, mask_zona_busqueda, mask_nucleo_final)
            
            # 2. Cálculos Médicos
            area_nucleo = cv2.countNonZero(mask_nucleo_final)
            area_citoplasma = cv2.countNonZero(mask_citoplasma)
            area_total = area_nucleo + area_citoplasma
            
            if area_total == 0 or area_nucleo == 0:
                continue # Salteamos si no hay célula clara
                
            relacion_nc = area_nucleo / area_total
            mask_celula_completa = cv2.bitwise_or(mask_nucleo_final, mask_citoplasma)
            porcentaje_rosado = calcular_porcentaje_rosado_total(img_original, mask_celula_completa)
            
            gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
            pixeles_nucleo = gray[mask_nucleo_final > 0]
            rugosidad = round(float(np.std(pixeles_nucleo)), 2) if len(pixeles_nucleo) > 0 else 0.0
            
            # 3. Clasificación V4
            caracteristicas = {
                'Porcentaje_Rosado': porcentaje_rosado,
                'Rugosidad_Nucleo': rugosidad,
                'Relacion_NC': relacion_nc,
                'Area_Nucleo': area_nucleo
            }
            diagnostico = clasificar_celula_v4(caracteristicas)
            
            # Guardamos el resultado
            resultados.append({
                'Archivo': p.name,
                'Diagnostico': diagnostico
            })
            
        except Exception as e:
            # Si una foto falla, la salteamos para que el lote no se corte
            continue

    return pd.DataFrame(resultados)