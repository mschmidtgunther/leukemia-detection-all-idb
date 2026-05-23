import cv2
import numpy as np

def cargar_imagen(ruta):
    """Carga la imagen y devuelve versiones BGR (para OpenCV) y RGB (para matplotlib)."""
    img_bgr = cv2.imread(ruta)
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen en la ruta: {ruta}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb


def aislar_celula(img_rgb, mask):
    """
    Multiplica la máscara por la imagen original para dejar solo la célula y el fondo negro.
    """
    # cv2.bitwise_and aplica la máscara a la imagen a color
    celula_aislada = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    return celula_aislada