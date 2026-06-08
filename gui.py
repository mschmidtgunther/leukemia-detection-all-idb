import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QDialog, QGridLayout, QFrame)
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt

# IMPORTAMOS TUS MÓDULOS ACTUALES
from main import main as ejecutar_pipeline_completo
from src.preprocessing import preprocesar_imagen
from src.classification import clasificar_celula_v4
from src.features import calcular_porcentaje_rosado_total
from src.segmentation import (
    segmentar_otsu_vs_watershed, 
    filtrar_linfocitos, 
    expandir_mascara_celula, 
    segmentar_nucleo_agresivo, 
    aislar_citoplasma_real
)

# ==============================================================================
# VENTANA SECUNDARIA: VISUALIZACIÓN DEL PASO A PASO
# ==============================================================================
class VentanaResultados(QDialog):
    def __init__(self, img_orig, img_prep, mask_celula, img_overlay, diagnostico, relacion_nc):
        super().__init__()
        self.setWindowTitle("Análisis Individual - Paso a Paso")
        self.setFixedSize(1000, 850)
        self.setStyleSheet("background-color: #f8f9fa;")

        layout_principal = QVBoxLayout()
        layout_principal.setContentsMargins(15, 15, 15, 15)
        grid = QGridLayout()
        grid.setSpacing(15)

        # Funciones auxiliares para convertir imágenes de OpenCV a PyQt
        lbl_orig = self.crear_label_imagen(img_orig, "1. Imagen Original (RGB)")
        lbl_prep = self.crear_label_imagen(img_prep, "2. Canal Preprocesado (L*a*b*)")
        lbl_mask = self.crear_label_imagen(mask_celula, "3. Máscara Segmentada Completa")
        lbl_overlay = self.crear_label_imagen(img_overlay, "4. Separación: Núcleo (Rojo) / Cito (Verde)")

        # Agregar a la cuadrícula (2x2)
        grid.addWidget(lbl_orig, 0, 0)
        grid.addWidget(lbl_prep, 0, 1)
        grid.addWidget(lbl_mask, 1, 0)
        grid.addWidget(lbl_overlay, 1, 1)


        # Panel inferior para el Diagnóstico
        panel_diag = QFrame()
        panel_diag.setStyleSheet("background-color: #ffffff; border-radius: 8px; border: 1px solid #dee2e6;")
        layout_diag = QVBoxLayout()
        
        lbl_titulo_diag = QLabel("RESULTADO DEL ANÁLISIS")
        lbl_titulo_diag.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl_titulo_diag.setAlignment(Qt.AlignCenter)
        lbl_titulo_diag.setStyleSheet("color: #495057; border: none;")

        # Colores según el diagnóstico
        color_diag = "#2f855a" if diagnostico == 'Benign' else "#c53030"
        texto_diag = "SANO (Benigno)" if diagnostico == 'Benign' else f"ANOMALÍA (Fase: {diagnostico.upper()})"
        
        lbl_res = QLabel(texto_diag)
        lbl_res.setFont(QFont("Segoe UI", 16, QFont.Bold))
        lbl_res.setAlignment(Qt.AlignCenter)
        lbl_res.setStyleSheet(f"color: {color_diag}; border: none;")

        lbl_metricas = QLabel(f"Relación Núcleo/Citoplasma (N/C): {relacion_nc:.4f}")
        lbl_metricas.setFont(QFont("Segoe UI", 11))
        lbl_metricas.setAlignment(Qt.AlignCenter)
        lbl_metricas.setStyleSheet("color: #495057; border: none;")

        layout_diag.addWidget(lbl_titulo_diag)
        layout_diag.addWidget(lbl_res)
        layout_diag.addWidget(lbl_metricas)
        panel_diag.setLayout(layout_diag)

        # Ensamblar layout principal
        layout_principal.addLayout(grid)
        layout_principal.addWidget(panel_diag)
        self.setLayout(layout_principal)

    def crear_label_imagen(self, cv_img, titulo):
        """Convierte matriz numpy (OpenCV) a QPixmap y la empaqueta con un título"""
        # Convertir a formato de PyQt
        if len(cv_img.shape) == 2:  # Escala de grises
            h, w = cv_img.shape
            qimg = QImage(cv_img.copy().data, w, h, w, QImage.Format_Grayscale8)
        else:  # RGB
            h, w, ch = cv_img.shape
            qimg = QImage(cv_img.copy().data, w, h, ch * w, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimg).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Crear contenedor
        container = QWidget()
        container.setStyleSheet("background-color: white; border-radius: 5px; border: 1px solid #ced4da;")
        lay = QVBoxLayout()
        
        lbl_title = QLabel(titulo)
        lbl_title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("border: none;")
        
        lbl_img = QLabel()
        lbl_img.setPixmap(pixmap)
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setStyleSheet("border: none;")
        
        lay.addWidget(lbl_title)
        lay.addWidget(lbl_img)
        container.setLayout(lay)
        return container

# ==============================================================================
# VENTANA PRINCIPAL DEL SOFTWARE
# ==============================================================================
class LeucemiaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAD - Leucemia Linfoblástica Aguda")
        self.setFixedSize(500, 350)
        
        # Estilo global de la ventana principal
        self.setStyleSheet("""
            QMainWindow { background-color: #f1f3f5; }
            QPushButton { 
                border-radius: 8px; 
                padding: 10px; 
                font-family: 'Segoe UI';
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { opacity: 0.8; }
        """)
        
        widget_central = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(25)
        
        # Título
        self.lbl_titulo = QLabel("Sistema de Análisis Hematológico")
        self.lbl_titulo.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.lbl_titulo.setAlignment(Qt.AlignCenter)
        self.lbl_titulo.setStyleSheet("color: #212529;")
        
        self.lbl_subtitulo = QLabel("Seleccione el modo de operación para iniciar")
        self.lbl_subtitulo.setFont(QFont("Segoe UI", 11))
        self.lbl_subtitulo.setAlignment(Qt.AlignCenter)
        self.lbl_subtitulo.setStyleSheet("color: #495057;")
        
        # Botones
        self.btn_individual = QPushButton("🔍 PROCESAR IMAGEN INDIVIDUAL (Paso a Paso)")
        self.btn_individual.setFixedSize(350, 55)
        self.btn_individual.setStyleSheet("background-color: #339af0; color: white; border: 1px solid #228be6;")
        self.btn_individual.clicked.connect(self.procesar_individual)
        self.btn_individual.setCursor(Qt.PointingHandCursor)
        
        self.btn_dataset = QPushButton("📂 PROCESAR BASE DE DATOS COMPLETA")
        self.btn_dataset.setFixedSize(350, 55)
        self.btn_dataset.setStyleSheet("background-color: #20c997; color: white; border: 1px solid #12b886;")
        self.btn_dataset.clicked.connect(self.procesar_dataset)
        self.btn_dataset.setCursor(Qt.PointingHandCursor)
        
        # Ensamblar
        layout.addWidget(self.lbl_titulo)
        layout.addWidget(self.lbl_subtitulo)
        layout.addSpacing(10)
        layout.addWidget(self.btn_individual, alignment=Qt.AlignCenter)
        layout.addWidget(self.btn_dataset, alignment=Qt.AlignCenter)
        
        widget_central.setLayout(layout)
        self.setCentralWidget(widget_central)

    def procesar_dataset(self):
        # Muestra alerta antes de congelar la ventana
        QMessageBox.information(self, "Iniciando", "Se iniciará la extracción masiva. Por favor, revise la consola y espere a que termine el procesamiento.")
        ejecutar_pipeline_completo()

    def procesar_individual(self):
        ruta_img, _ = QFileDialog.getOpenFileName(self, "Seleccionar Frotis Sanguíneo", "", "Imágenes (*.jpg *.png *.jpeg)")
        if not ruta_img:
            return 
            
        try:
            # 1. Pipeline de Procesamiento
            img_original, img_preprocesada = preprocesar_imagen(ruta_img)
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
                raise ValueError("No se detectó ninguna célula clara en la imagen. Verifique la calidad de la foto.")
                
            relacion_nc = area_nucleo / area_total
            mask_celula_completa = cv2.bitwise_or(mask_nucleo_final, mask_citoplasma)
            porcentaje_rosado = calcular_porcentaje_rosado_total(img_original, mask_celula_completa)
            
            gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
            pixeles_nucleo = gray[mask_nucleo_final > 0]
            rugosidad = round(float(np.std(pixeles_nucleo)), 2) if len(pixeles_nucleo) > 0 else 0.0
            
            # 3. Generación Visual: Overlay Núcleo/Citoplasma
            # Pintamos el núcleo de rojo semitransparente y citoplasma de verde
            img_overlay = img_original.copy()
            overlay_color = np.zeros_like(img_original)
            overlay_color[mask_nucleo_final > 0] = [255, 0, 0]    # Rojo
            overlay_color[mask_citoplasma > 0] = [0, 255, 0]      # Verde
            cv2.addWeighted(overlay_color, 0.4, img_overlay, 0.6, 0, img_overlay)
            
            # 4. Clasificación V4
            caracteristicas = {
                'Porcentaje_Rosado': porcentaje_rosado,
                'Rugosidad_Nucleo': rugosidad,
                'Relacion_NC': relacion_nc,
                'Area_Nucleo': area_nucleo
            }
            diagnostico = clasificar_celula_v4(caracteristicas)
            
            # 5. Desplegar Ventana de Resultados
            self.ventana_res = VentanaResultados(
                img_orig=img_original,
                img_prep=img_preprocesada,
                mask_celula=mask_celula_completa,
                img_overlay=img_overlay,
                diagnostico=diagnostico,
                relacion_nc=relacion_nc
            )
            self.ventana_res.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Error de Procesamiento", f"Ocurrió un error al analizar la imagen:\n{str(e)}")

if __name__ == "__main__":
    # Ajuste de escalado de alta definición para pantallas modernas
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = LeucemiaApp()
    ventana.show()
    sys.exit(app.exec_())