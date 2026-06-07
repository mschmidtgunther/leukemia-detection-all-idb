"Adquisicion de datos y preprocesamiento"

import cv2


## Obtener las rutas de las imágenes

def obtener_rutas_imagenes(raw_path, clases):
    """
    Carga todas las imágenes válidas de data/raw/Original
    
    """
    dataset = {}
    for cls in clases:
        carpeta = raw_path / cls
        imgs = list(carpeta.glob('*.jpg')) + list(carpeta.glob('*.png'))
        imgs_validas = []
        for p in imgs:
            img = cv2.imread(str(p))
            # Verificamos que la imagen se lea bien y no esté corrupta/negra
            if img is not None and img.mean() >= 5:
                imgs_validas.append(p)
        dataset[cls] = imgs_validas

    todas = [p for paths in dataset.values() for p in paths]

    print('Dataset cargado:')
    for cls, imgs in dataset.items():
        print(f'  {cls:<10}: {len(imgs)} imágenes')
    print(f'  {"Total":<10}: {len(todas)} imágenes')

    return dataset, todas


# Preprocesamiento de imágenes

def preprocesar_imagen(ruta):
    """
    Toma la ruta de una imagen, la lee, cambia de espacio de color, 
    reduce el ruido y mejora el contraste del núcleo.
    
    """
    # 1. Leer imagen
    img_bgr = cv2.imread(str(ruta))
    # Convertir a RGB para que matplotlib la muestre con los colores reales
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Convertir a espacio de color CIE L*a*b*
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # Separar los canales L, a, b. 
    # El canal 'b*' resalta muy bien los núcleos púrpuras/azules frente al fondo
    l, a, b = cv2.split(img_lab)
    
    # 3. Filtrado de ruido con Filtro de Mediana
    # Suaviza el interior de la célula sin difuminar los bordes (kernel 5x5)
    b_suavizado = cv2.medianBlur(b, 5)
    
    # 4. Mejora de Contraste con CLAHE
    # Ayuda a homogeneizar la iluminación y marcar mejor los límites
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    canal_listo = clahe.apply(b_suavizado)
    
    return img_rgb, canal_listo



