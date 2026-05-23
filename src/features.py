import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clasificar_y_filtrar_linfocitos(mascara_celula, mascara_nucleo, imagen_original):
    """Filtra objetos basándose en la forma de la célula y, 
    en la circularidad de su núcleo para descartar neutrófilos u otras células no deseadas."""

    contornos_celula, _ = cv2.findContours(mascara_celula, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mascara_linfocitos = np.zeros_like(mascara_celula)
    caracteristicas_extraidas = []
    
    for cnt in contornos_celula:
        area_celula = cv2.contourArea(cnt)
        
        # 1. Filtro de Área: Bajamos el mínimo a 300 por si la célula del medio era pequeña
        if area_celula < 400 or area_celula > 80000:
            continue
            
        # 2. Solidez de la célula completa (bajamos a 0.75 para ser más permisivos)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidez = area_celula / float(hull_area) if hull_area > 0 else 0
        
        # --- 3. ANÁLISIS DEL NÚCLEO ---
        # Aislamos el núcleo que pertenece ÚNICAMENTE a esta célula actual
        mask_esta_celula = np.zeros_like(mascara_celula)
        cv2.drawContours(mask_esta_celula, [cnt], -1, 255, thickness=cv2.FILLED)
        
        nucleo_aislado = cv2.bitwise_and(mascara_nucleo, mascara_nucleo, mask=mask_esta_celula)
        contornos_nuc, _ = cv2.findContours(nucleo_aislado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contornos_nuc) == 0:
            continue # Si no tiene núcleo, es basura
            
        # Tomamos el núcleo principal de esta célula
        cnt_nucleo = max(contornos_nuc, key=cv2.contourArea)
        area_nuc = cv2.contourArea(cnt_nucleo)
        peri_nuc = cv2.arcLength(cnt_nucleo, True)
        
        # Circularidad del Núcleo
        circularidad_nucleo = 0
        if peri_nuc > 0:
            circularidad_nucleo = 4 * np.pi * (area_nuc / (peri_nuc * peri_nuc))
            
        # --- REGLA DE ORO (Clasificador) ---
        # El núcleo DEBE ser redondo (> 0.55) y la célula medianamente sólida (> 0.75) Se pueden tantear los valores para mejorar
        es_linfocito = (circularidad_nucleo > 0.75) and (solidez > 0.75)
        
        if es_linfocito:
            cv2.drawContours(mascara_linfocitos, [cnt], -1, 255, thickness=cv2.FILLED)
            caracteristicas_extraidas.append({
                'Area_Celula': area_celula,
                'Area_Nucleo': area_nuc,
                'Solidez_Celula': solidez,
                'Circularidad_Nucleo': circularidad_nucleo
            })
            
    imagen_solo_linfocitos = cv2.bitwise_and(imagen_original, imagen_original, mask=mascara_linfocitos)
    
    return mascara_linfocitos, imagen_solo_linfocitos, caracteristicas_extraidas

