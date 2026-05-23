import cv2

def preprocesamiento(img_path, filtro='mediana', ksize=5):
    """
    Pipeline completo de preprocesamiento: 
        1. Carga imagen
        2. Convierte a HSV y extrae canal S
        3. Aplica filtro de ruido mediana
    Retorna: imagen original RGB, canal S sin procesar, canal S preprocesado
    """
    img_bgr = cv2.imread(str(img_path))
    
    # Seguro por si alguna ruta se rompe en otra compu
    if img_bgr is None:
        raise FileNotFoundError(f"No se encontró la imagen en: {img_path}")
    

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    canal_s = img_hsv[:,:,1] # Canal S (Saturación - Índice 1)

    if filtro == 'mediana':
        canal_filtrado = cv2.medianBlur(canal_s, ksize)
    elif filtro == 'gaussiano':
        canal_filtrado = cv2.GaussianBlur(canal_s, (ksize, ksize), 0)
    else:
        canal_filtrado = canal_s
        print(f"Filtro '{filtro}' no reconocido. Se devuelve el canal S limpio.")
    
    return img_rgb, canal_s, canal_filtrado