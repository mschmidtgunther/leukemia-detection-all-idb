import cv2
import numpy as np


def segmentar_nucleo_estricto(canal_proc):
    """Genera la máscara marcando solo los núcleos usando el método de Otsu original.
    Ideal para medir luego la circularidad del núcleo y descartar neutrófilos.
    """
    # 1. Otsu matemático normal (encuentra lo más oscuro/saturado por sí solo)
    umbral_otsu, mascara_nucleo = cv2.threshold(
        canal_proc, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # 2. Pequeño cierre morfológico para que el núcleo no tenga "agujeros" negros adentro
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_nucleo_limpia = cv2.morphologyEx(mascara_nucleo, cv2.MORPH_CLOSE, kernel)
    
    return mascara_nucleo_limpia

def segmentar_celula_otsu(canal_proc, umbral_maximo=10):
    """
    Segmenta la célula completa (núcleo + citoplasma) usando Otsu.
    Usa un 'techo' de seguridad: si Otsu se va muy arriba y solo agarra 
    el núcleo, lo forzamos a bajar para incluir el citoplasma.
    """
    # 1. Calculamos Otsu estándar
    umbral_otsu, mascara_otsu = cv2.threshold(
        canal_proc, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # 2. Control de seguridad INVERTIDO: 
    # Si Otsu calculó un umbral muy alto (ej. 100), perdemos el citoplasma. 
    # Lo forzamos a bajar al umbral_maximo.
    if umbral_otsu > umbral_maximo:
        # Re-calculamos usando el umbral fijo más bajo
        _, mascara = cv2.threshold(canal_proc, umbral_maximo, 255, cv2.THRESH_BINARY)
        umbral_final = umbral_maximo
    else:
        mascara = mascara_otsu
        umbral_final = umbral_otsu
        
    return mascara, umbral_final

def refinar_mascara_nucleo(mascara, kernel_size=5):
    """
    Aplica operaciones morfológicas para limpiar la máscara del núcleo:
    1. Apertura: elimina pequeños artefactos y puentes finos.
    2. Cierre: rellena huecos internos en el núcleo.
    3. Relleno de agujeros con SciPy: consolida la máscara final.
    """
    # Usamos un elemento estructurante elíptico (ideal para formas redondeadas de núcleos)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 1. Apertura (Erosión + Dilatación) - Limpia ruidos externos
    apertura = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    
    # 2. Cierre (Dilatación + Erosión) - Conecta componentes del mismo núcleo
    cierre = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)
    
    # 3. Relleno de agujeros internos (SciPy ndimage)
    relleno = ndimage.binary_fill_holes(cierre).astype(np.uint8) * 255
    
    # Retornamos los pasos intermedios para tu gráfico de visualización, 
    # siendo 'relleno' el resultado final optimizado.
    return apertura, cierre, relleno
