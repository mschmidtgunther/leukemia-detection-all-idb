"Extracción de características de las imágenes utilizando técnicas clásicas de procesamiento de imágenes. Estas funciones se pueden usar para generar un conjunto de características que luego pueden ser alimentadas a un modelo de machine learning para clasificación o detección."

import cv2
import numpy as np
import pandas as pd

# Importamos las funciones que ya tienes definidas en tu pipeline de imágenes
# (Asegúrate de que estas funciones estén declaradas arriba en el mismo archivo o importadas correctamente)
from src.preprocessing import preprocesar_imagen
from src.segmentation import (
    segmentar_otsu_vs_watershed, 
    filtrar_linfocitos, 
    expandir_mascara_celula, 
    segmentar_nucleo_agresivo, 
    aislar_citoplasma_real
)

# Función para calcular el porcentaje de tonos rosados en toda la célula

def calcular_porcentaje_rosado_total(img_rgb, mask_celula):
    if cv2.countNonZero(mask_celula) == 0: return 0.0
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    rango_bajo = np.array([125, 20, 40])
    rango_alto = np.array([175, 255, 255])
    mask_rosado = cv2.inRange(hsv, rango_bajo, rango_alto)
    rosado_celula = cv2.bitwise_and(mask_rosado, mask_celula)
    return round(cv2.countNonZero(rosado_celula) / cv2.countNonZero(mask_celula), 4)

def extract_dataset_features(dataset):
    """
    Recorre el diccionario del dataset, extrae las características geométricas,
    colorimétricas y texturales (incluyendo Rugosidad), y devuelve un DataFrame limpio.
    """
    datos_extraccion = []

    print("Extrayendo características reales de todo el dataset...")

    for clase, rutas in dataset.items():
        for i, ruta in enumerate(rutas):
            try:
                # 1. Preprocesamiento
                img_original, img_preprocesada = preprocesar_imagen(ruta)
                
                # 2. Máscara Base (Watershed)
                _, mask_ws = segmentar_otsu_vs_watershed(img_original, img_preprocesada, fg_factor=0.52)
                mask_base = filtrar_linfocitos(mask_ws, min_area=200, min_circularidad=0.75)
                
                # 3. Zona de búsqueda
                mask_zona_busqueda = expandir_mascara_celula(mask_base, iteraciones=1)
                
                # 4. Núcleo (Uso un Otsu estándar equilibrado, sin extremos)
                mask_nucleo_bruto = segmentar_nucleo_agresivo(img_preprocesada, ajuste_otsu=-10)
                mask_nucleo_final = cv2.bitwise_and(mask_nucleo_bruto, mask_zona_busqueda)
                
                # 5. Citoplasma
                mask_citoplasma = aislar_citoplasma_real(img_original, mask_zona_busqueda, mask_nucleo_final)
                
                # Cantidad de píxeles
                area_nucleo = cv2.countNonZero(mask_nucleo_final)
                area_citoplasma = cv2.countNonZero(mask_citoplasma)
                area_total = area_nucleo + area_citoplasma
                
                if area_total > 0:
                    relacion_nc = area_nucleo / area_total
                    mask_celula_completa = cv2.bitwise_or(mask_nucleo_final, mask_citoplasma)
                    porcentaje_rosado = calcular_porcentaje_rosado_total(img_original, mask_celula_completa)
                    
                    # -----------------------------------------------------------------
                    # NUEVA CARACTERÍSTICA: Rugosidad del Núcleo
                    # -----------------------------------------------------------------
                    # Pasamos la imagen original a escala de grises para medir la textura del núcleo
                    gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
                    # Extraemos únicamente los píxeles que pertenecen al núcleo analizado
                    pixeles_nucleo = gray[mask_nucleo_final > 0]
                    
                    # Calculamos la rugosidad como la desviación estándar de la intensidad de grises
                    if len(pixeles_nucleo) > 0:
                        rugosidad = round(float(np.std(pixeles_nucleo)), 2)
                    else:
                        rugosidad = 0.0
                    # -----------------------------------------------------------------
                    
                    # Guardamos los datos crudos incluyendo la variable corregida
                    datos_extraccion.append({
                        "Clase": clase,
                        "Area_Nucleo": area_nucleo,
                        "Area_Citoplasma": area_citoplasma,
                        "Relacion_NC": round(relacion_nc, 4),
                        "Porcentaje_Rosado": porcentaje_rosado,
                        "Rugosidad_Nucleo": rugosidad  # <-- ¡Inyectada con éxito!
                    })
            except Exception as e:
                # Si alguna imagen falla o está corrupta, la salteamos silenciosamente
                continue

    # Convertimos a un DataFrame de Pandas
    df_caracteristicas = pd.DataFrame(datos_extraccion)
    return df_caracteristicas