"Segmentación de imágenes"

import cv2
import numpy as np

# Segmentar con Otsu y Watershed, permitiendo ajustar la barrera de Watershed con fg_factor

def segmentar_otsu_vs_watershed(img_rgb, canal_preprocesado, fg_factor=0.6):
    """
    Aplica segmentación. Permite ajustar qué tan estricto es Watershed 
    para definir el centro "seguro" del núcleo a través de fg_factor.
    
    """
    # ---------------------------------------------------------
    # 1. MÉTODO OTSU (Lo dejamos limpio, solo para generar la base)
    # ---------------------------------------------------------
    _, mask_otsu = cv2.threshold(canal_preprocesado, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel, iterations=2)   # Elimina pequeñas manchas (ruido)
    mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)  # Cierra pequeños agujeros dentro de las regiones segmentadas

    # ---------------------------------------------------------
    # 2. MÉTODO WATERSHED CON BARRERA AJUSTABLE
    # ---------------------------------------------------------
    # Fondo seguro (dilatamos la máscara base)
    sure_bg = cv2.dilate(mask_otsu, kernel, iterations=3)
    
    # Transformada de distancia (encuentra los "centros" más gruesos)
    dist_transform = cv2.distanceTransform(mask_otsu, cv2.DIST_L2, 5)  
    
    # Aplicamos barrera (fg_factor). 
    # Multiplicamos el valor máximo por este factor.
    # Valores más cercanos a 1.0 son EXTREMADAMENTE estrictos.
    _, sure_fg = cv2.threshold(dist_transform, fg_factor * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Región dudosa
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Etiquetamos los marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 
    markers[unknown == 255] = 0 
    
    # Aplicamos Watershed
    img_ws = img_rgb.copy()
    markers = cv2.watershed(img_ws, markers)
    
    # Reconstruimos la máscara
    mask_watershed = np.zeros_like(canal_preprocesado, dtype=np.uint8)
    mask_watershed[markers > 1] = 255 
    
    return mask_otsu, mask_watershed


# Extraer la región segmentada con sus colores originales

def extraer_region_color(img_rgb, mascara):
    """
    Toma la imagen original y la máscara binaria.
    Multiplica ambas para 'apagar' el fondo (negro) y dejar solo 
    la región de interés con sus colores originales.
    """
    # cv2.bitwise_and aplica una operación AND a nivel de bits.
    # Al pasarle la máscara, solo mantiene los píxeles donde la máscara es > 0.
    nucleo_extraido = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara)
    
    return nucleo_extraido


# Filtrar linfocitos por área y circularidad

def filtrar_linfocitos(mascara, min_area=1000, min_circularidad=0.5):
    """
    Busca los contornos en la máscara y filtra aquellos que no cumplen
    con el tamaño o la forma típica de un núcleo de linfocito.
    """
    # Encontramos los contornos en la máscara de Watershed
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Creamos una máscara negra vacía para dibujar solo lo que pase el filtro
    mascara_filtrada = np.zeros_like(mascara)
    
    for c in contornos:
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)
        
        # Evitar división por cero
        if perimetro == 0:
            continue
            
        # Calculamos la circularidad
        circularidad = 4 * np.pi * (area / (perimetro * perimetro))
        
        # Filtramos por Área y Circularidad
        if area > min_area and circularidad > min_circularidad:
            # Si cumple las condiciones, lo dibujamos en blanco (255) en la nueva máscara
            # El -1 indica que rellene todo el contorno
            cv2.drawContours(mascara_filtrada, [c], -1, 255, -1)
            
    return mascara_filtrada


# Dilatar la máscara del linfocito para asegurar que capture el citoplasma

def expandir_mascara_celula(mascara_base, iteraciones=1):
    """
    Toma la máscara del linfocito ya filtrada y la dilata (expande) 
    artificialmente para asegurar que englobe todo el citoplasma.
    """
    kernel = np.ones((5,5), np.uint8)
    mascara_expandida = cv2.dilate(mascara_base, kernel, iterations=iteraciones)
    return mascara_expandida


# Segmentar el núcleo agresivamente para asegurar que capture incluso los núcleos pálidos (Pro)

def segmentar_nucleo_agresivo(canal_preprocesado, ajuste_otsu=-20):
    """
    Detecta el núcleo estrictamente por color oscuro (para salvar el citoplasma Benign), 
    pero aplica un pegamento morfológico elíptico para fusionar núcleos pálidos (Pro).
    """
    # 1. Umbral MUY estricto
    umbral_otsu, _ = cv2.threshold(canal_preprocesado, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    umbral_estricto = umbral_otsu + ajuste_otsu
    _, mask_nucleo = cv2.threshold(canal_preprocesado, umbral_estricto, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Pegamento morfológico elíptico (Más orgánico que el cuadrado)
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_nucleo = cv2.morphologyEx(mask_nucleo, cv2.MORPH_CLOSE, kernel_cierre)
    
    # Pequeña limpieza de bordes
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_nucleo = cv2.morphologyEx(mask_nucleo, cv2.MORPH_OPEN, kernel_apertura)
    
    # 3. Rellenado de contornos (Rellena la fase Pro si quedó hueca)
    contornos, _ = cv2.findContours(mask_nucleo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_rellena = np.zeros_like(mask_nucleo)
    
    for c in contornos:
        # Solo conservamos aglomeraciones grandes de núcleo (evita ruido)
        if cv2.contourArea(c) > 150:
            cv2.drawContours(mask_rellena, [c], -1, 255, -1)
            
    return mask_rellena

# Finalmente, usamos el canal de Saturación para atrapar el citoplasma pálido de las células benignas

def aislar_citoplasma_real(img_rgb, mascara_expandida, mascara_nucleo):
    """
    Usa el canal de Saturación con un umbral relajado para atrapar 
    el citoplasma pálido de las células benignas.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    _, s, _ = cv2.split(hsv)
    
    # 1. Otsu sobre la saturación, pero le restamos 15 puntos para ser permisivos
    umbral_otsu, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    umbral_relajado = max(0, umbral_otsu - 15)  # Evitamos que baje de 0
    _, mascara_materia = cv2.threshold(s, umbral_relajado, 255, cv2.THRESH_BINARY)
    
    # 2. Intersección
    celula_real = cv2.bitwise_and(mascara_materia, mascara_expandida)
    
    # 3. FILTRO TOPOLÓGICO (Fusión y Limpieza)
    contornos, _ = cv2.findContours(celula_real, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    celula_limpia = np.zeros_like(celula_real)
    
    for c in contornos:
        temp_mask = np.zeros_like(celula_real)
        cv2.drawContours(temp_mask, [c], -1, 255, -1)
        
        superposicion = cv2.bitwise_and(temp_mask, mascara_nucleo)
        
        if cv2.countNonZero(superposicion) > 0:
            cv2.drawContours(celula_limpia, [c], -1, 255, -1)
            
    # 4. Restamos el núcleo
    citoplasma_final = cv2.subtract(celula_limpia, mascara_nucleo)
    
    return citoplasma_final

