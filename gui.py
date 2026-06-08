"Interfaz grafica"

import sys
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QMessageBox, QDialog, QGridLayout, QFrame, QScrollArea,
                            QSizePolicy)
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt

# Librerías para incrustar gráficos de Matplotlib en la interfaz
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

# IMPORTAMOS TU MÓDULO PARA PROCESAR LOTES
from src.multclasificador import procesar_lote_ciego

# ==============================================================================
# RESULTADOS DEL LOTE (GRÁFICOS DE BARRAS)
# ==============================================================================
class VentanaReporteLote(QDialog):
    def __init__(self, df_resultados):
        super().__init__()
        self.setWindowTitle("Reporte Epidemiológico de Clasificación")
        self.setFixedSize(900, 500)
        self.setStyleSheet("background-color: #ffffff;")

        layout_principal = QVBoxLayout()
        
        lbl_titulo = QLabel(f"Resumen de Procesamiento: {len(df_resultados)} células analizadas")
        lbl_titulo.setFont(QFont("Segoe UI", 16, QFont.Bold))
        lbl_titulo.setAlignment(Qt.AlignCenter)
        lbl_titulo.setStyleSheet("color: #2b8a3e; margin-top: 10px;")
        layout_principal.addWidget(lbl_titulo)

        # Contenedor para los gráficos de Matplotlib
        self.canvas = FigureCanvas(Figure(figsize=(10, 5)))
        layout_principal.addWidget(self.canvas)
        
        self.setLayout(layout_principal)
        
        # Generar los gráficos de barras
        self.dibujar_graficos(df_resultados)

    def dibujar_graficos(self, df):
        ax1 = self.canvas.figure.add_subplot(121)
        ax2 = self.canvas.figure.add_subplot(122)

        total = len(df)
        sanas = len(df[df['Diagnostico'] == 'Benign'])
        cancerigenas = total - sanas

        labels_1 = ['Sanas\n(Benign)', 'Anomalías\n(LLA)']
        counts_1 = [sanas, cancerigenas]
        colores_1 = ['#40c057', '#fa5252']
        
        bars1 = ax1.bar(labels_1, counts_1, color=colores_1, edgecolor='black', zorder=3)
        ax1.set_title("Proporción Global Detectada", fontweight="bold", pad=15)
        ax1.set_ylabel("Cantidad de Células", fontweight="bold")
        ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

        for bar, count in zip(bars1, counts_1):
            if total > 0:
                porcentaje = (count / total) * 100
                texto = f'{count}\n({porcentaje:.1f}%)'
            else:
                texto = '0\n(0%)'
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), texto, 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
                    
        limite_y1 = max(counts_1) * 1.2 if max(counts_1) > 0 else 10
        ax1.set_ylim(0, limite_y1)

        if cancerigenas > 0:
            df_enfermas = df[df['Diagnostico'] != 'Benign']
            conteo = df_enfermas['Diagnostico'].value_counts()
            
            labels_2 = conteo.index.tolist()
            counts_2 = conteo.values.tolist()
            colores_2 = ['#fd7e14', '#f59f00', '#f03e3e']
            
            bars2 = ax2.bar(labels_2, counts_2, color=colores_2[:len(labels_2)], edgecolor='black', zorder=3)
            ax2.set_title("Desglose por Fases de LLA", fontweight="bold", pad=15)
            ax2.set_ylabel("Cantidad de Células", fontweight="bold")
            ax2.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

            for bar, count in zip(bars2, counts_2):
                porcentaje = (count / cancerigenas) * 100
                texto = f'{count}\n({porcentaje:.1f}%)'
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), texto, 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
                        
            ax2.set_ylim(0, max(counts_2) * 1.2)
        else:
            ax2.text(0.5, 0.5, 'No se detectaron células\ncancerígenas en la muestra.', 
                    horizontalalignment='center', verticalalignment='center', 
                    fontsize=12, fontweight='bold', color='grey')
            ax2.axis('off')

        self.canvas.figure.tight_layout()
        self.canvas.draw()


# ==============================================================================
# INSTRUCCIONES PARA MÚLTIPLES IMÁGENES 
# ==============================================================================
class DialogoInstrucciones(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Instrucciones - Múltiples Imágenes")
        # Sin setFixedSize: el diálogo se ajusta al contenido automáticamente
        self.setMinimumWidth(520)
        self.setStyleSheet("background-color: #f8f9fa;")

        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(18)

        lbl_titulo = QLabel("Requisitos para la Clasificación en Lote")
        lbl_titulo.setFont(QFont("Segoe UI", 13, QFont.Bold))
        lbl_titulo.setAlignment(Qt.AlignCenter)
        lbl_titulo.setStyleSheet("color: #212529;")

        lbl_info = QLabel(
            "Para analizar múltiples imágenes en lote, asegúrese de cumplir con:\n\n"
            "  1.  Todos los archivos deben estar en formato  .jpg  o  .png\n\n"
            "  2.  Las imágenes deben estar ubicadas en la siguiente ruta:\n\n"
            "           ./data/clasificacion"
        )
        lbl_info.setFont(QFont("Segoe UI", 10))
        lbl_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lbl_info.setStyleSheet("color: #495057; background-color: #ffffff; "
                            "border: 1px solid #dee2e6; border-radius: 6px; padding: 12px;")
        lbl_info.setWordWrap(True)
        # El label crece tanto como necesite verticalmente
        lbl_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        btn_aceptar = QPushButton("Aceptar e Iniciar Clasificación")
        btn_aceptar.setFixedSize(270, 42)
        btn_aceptar.setFont(QFont("Segoe UI", 10, QFont.Bold))
        btn_aceptar.setStyleSheet("background-color: #f59f00; color: white; border-radius: 6px;")
        btn_aceptar.setCursor(Qt.PointingHandCursor)
        btn_aceptar.clicked.connect(self.accept)

        layout.addWidget(lbl_titulo)
        layout.addWidget(lbl_info)
        layout.addWidget(btn_aceptar, alignment=Qt.AlignCenter)
        self.setLayout(layout)
        # Ajusta el tamaño de la ventana al contenido real
        self.adjustSize()


# ==============================================================================
# VISUALIZACIÓN DEL PASO A PASO  
# ==============================================================================
# Tamaño de cada imagen. 
IMG_SIZE = 300

class VentanaResultados(QDialog):
    def __init__(self, img_orig, img_prep, mask_celula, img_overlay, diagnostico, relacion_nc):
        super().__init__()
        self.setWindowTitle("Análisis Individual - Paso a Paso")

        # Ventana calculada en base a IMG_SIZE para que siempre encaje justo
        ancho = (IMG_SIZE + 40) * 2 + 60   # 2 columnas + márgenes
        alto  = (IMG_SIZE + 45) * 2 + 130  # 2 filas + panel diagnóstico
        self.setFixedSize(ancho, alto)
        self.setStyleSheet("background-color: #f8f9fa;")

        layout_principal = QVBoxLayout()
        layout_principal.setContentsMargins(20, 20, 20, 20)
        layout_principal.setSpacing(15)

        grid = QGridLayout()
        grid.setSpacing(15)
        # Cada columna y fila crece por igual
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        lbl_orig    = self.crear_label_imagen(img_orig,    "1. Imagen Original (RGB)")
        lbl_prep    = self.crear_label_imagen(img_prep,    "2. Canal Preprocesado (L*a*b*)")
        lbl_mask    = self.crear_label_imagen(mask_celula, "3. Máscara Segmentada Completa")
        lbl_overlay = self.crear_label_imagen(img_overlay, "4. Separación: Núcleo (Rojo) / Cito (Verde)")

        grid.addWidget(lbl_orig,    0, 0)
        grid.addWidget(lbl_prep,    0, 1)
        grid.addWidget(lbl_mask,    1, 0)
        grid.addWidget(lbl_overlay, 1, 1)

        # ── Panel de diagnóstico ──────────────────────────────────────────────
        panel_diag = QFrame()
        panel_diag.setFixedHeight(105)
        panel_diag.setStyleSheet("background-color: #ffffff; border-radius: 8px; border: 1px solid #dee2e6;")
        layout_diag = QVBoxLayout()
        layout_diag.setContentsMargins(15, 10, 15, 10)
        layout_diag.setSpacing(4)
        
        lbl_titulo_diag = QLabel("RESULTADO DEL ANÁLISIS")
        lbl_titulo_diag.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_titulo_diag.setAlignment(Qt.AlignCenter)
        lbl_titulo_diag.setStyleSheet("color: #6c757d; border: none;")

        color_diag = "#2f855a" if diagnostico == 'Benign' else "#c53030"
        texto_diag = "SANO (Benigno)" if diagnostico == 'Benign' else f"ANOMALÍA  —  Fase: {diagnostico.upper()}"
        
        lbl_res = QLabel(texto_diag)
        lbl_res.setFont(QFont("Segoe UI", 16, QFont.Bold))
        lbl_res.setAlignment(Qt.AlignCenter)
        lbl_res.setStyleSheet(f"color: {color_diag}; border: none;")

        lbl_metricas = QLabel(f"Relación Núcleo / Citoplasma (N/C):  {relacion_nc:.4f}")
        lbl_metricas.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_metricas.setAlignment(Qt.AlignCenter)
        lbl_metricas.setStyleSheet("color: #495057; border: none;")

        layout_diag.addWidget(lbl_titulo_diag)
        layout_diag.addWidget(lbl_res)
        layout_diag.addWidget(lbl_metricas)
        panel_diag.setLayout(layout_diag)

        layout_principal.addLayout(grid)
        layout_principal.addWidget(panel_diag)
        self.setLayout(layout_principal)

    def crear_label_imagen(self, cv_img, titulo):
        """Convierte matriz numpy (OpenCV) a QPixmap y la empaqueta con un título."""
        if len(cv_img.shape) == 2:          # escala de grises
            h, w = cv_img.shape
            qimg = QImage(cv_img.copy().data, w, h, w, QImage.Format_Grayscale8)
        else:                               # RGB
            h, w, ch = cv_img.shape
            qimg = QImage(cv_img.copy().data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(
            IMG_SIZE, IMG_SIZE,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Contenedor de tamaño fijo: imagen + título
        container = QFrame()
        container.setFixedSize(IMG_SIZE + 30, IMG_SIZE + 38)
        container.setStyleSheet("background-color: white; border-radius: 6px; border: 1px solid #ced4da;")

        lay = QVBoxLayout()
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(5)

        lbl_title = QLabel(titulo)
        lbl_title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("border: none; color: #212529;")

        lbl_img = QLabel()
        lbl_img.setPixmap(pixmap)
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setFixedSize(IMG_SIZE, IMG_SIZE)
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
        self.setFixedSize(500, 420) 
        
        self.setStyleSheet("""
            QMainWindow { background-color: #f1f3f5; }
            QPushButton { 
                border-radius: 8px; 
                padding: 10px; 
                font-family: 'Segoe UI';
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { opacity: 0.9; }
        """)
        
        widget_central = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        
        self.lbl_titulo = QLabel("Sistema de Análisis Hematológico")
        self.lbl_titulo.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.lbl_titulo.setAlignment(Qt.AlignCenter)
        self.lbl_titulo.setStyleSheet("color: #212529;")
        
        self.lbl_subtitulo = QLabel("Seleccione el modo de operación para iniciar")
        self.lbl_subtitulo.setFont(QFont("Segoe UI", 11))
        self.lbl_subtitulo.setAlignment(Qt.AlignCenter)
        self.lbl_subtitulo.setStyleSheet("color: #495057;")

        # Estan re lindos los emojis si no queda re fea la interfaz, pero si querés sacarlos no hay drama.

        self.btn_individual = QPushButton("🔍 PROCESAR IMAGEN INDIVIDUAL (Paso a Paso)")
        self.btn_individual.setFixedSize(350, 55)
        self.btn_individual.setStyleSheet("background-color: #339af0; color: white; border: 1px solid #228be6;")
        self.btn_individual.clicked.connect(self.procesar_individual)
        self.btn_individual.setCursor(Qt.PointingHandCursor)

        self.btn_multiples = QPushButton("📁 CLASIFICAR MÚLTIPLES IMÁGENES")
        self.btn_multiples.setFixedSize(350, 55)
        self.btn_multiples.setStyleSheet("background-color: #f59f00; color: white; border: 1px solid #f08c00;")
        self.btn_multiples.clicked.connect(self.iniciar_clasificacion_multiples)
        self.btn_multiples.setCursor(Qt.PointingHandCursor)
        
        self.btn_dataset = QPushButton("📊 EVALUACIÓN CON BASE DE DATOS")
        self.btn_dataset.setFixedSize(350, 55)
        self.btn_dataset.setStyleSheet("background-color: #20c997; color: white; border: 1px solid #12b886;")
        self.btn_dataset.clicked.connect(self.procesar_dataset)
        self.btn_dataset.setCursor(Qt.PointingHandCursor)
        
        layout.addWidget(self.lbl_titulo)
        layout.addWidget(self.lbl_subtitulo)
        layout.addSpacing(10)
        layout.addWidget(self.btn_individual, alignment=Qt.AlignCenter)
        layout.addWidget(self.btn_multiples,  alignment=Qt.AlignCenter)
        layout.addWidget(self.btn_dataset,    alignment=Qt.AlignCenter)
        
        widget_central.setLayout(layout)
        self.setCentralWidget(widget_central)

    def procesar_dataset(self):
        QMessageBox.information(self, "Iniciando", "Se iniciará la extracción masiva. Por favor, revise la consola y espere a que termine el procesamiento.")
        ejecutar_pipeline_completo()

    def iniciar_clasificacion_multiples(self):
        dialogo = DialogoInstrucciones(self)
        if dialogo.exec_() == QDialog.Accepted:
            ruta_objetivo = "./data/clasificacion"
            
            if not os.path.exists(ruta_objetivo):
                QMessageBox.warning(self, "Error de Ruta", f"No se encontró la carpeta:\n{ruta_objetivo}\n\nPor favor, créela y coloque las imágenes allí.")
                return

            df_resultados = procesar_lote_ciego(ruta_objetivo)
            
            if df_resultados.empty:
                QMessageBox.warning(self, "Error", "La carpeta está vacía o las imágenes no pudieron ser procesadas.")
                return
            
            self.ventana_lote = VentanaReporteLote(df_resultados)
            self.ventana_lote.exec_()

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
            
            # 3. Overlay Núcleo/Citoplasma
            img_overlay = img_original.copy()
            overlay_color = np.zeros_like(img_original)
            overlay_color[mask_nucleo_final > 0] = [255, 0, 0]
            overlay_color[mask_citoplasma > 0]   = [0, 255, 0]
            cv2.addWeighted(overlay_color, 0.4, img_overlay, 0.6, 0, img_overlay)
            
            # 4. Clasificación V4
            caracteristicas = {
                'Porcentaje_Rosado': porcentaje_rosado,
                'Rugosidad_Nucleo':  rugosidad,
                'Relacion_NC':       relacion_nc,
                'Area_Nucleo':       area_nucleo
            }
            diagnostico = clasificar_celula_v4(caracteristicas)
            
            # 5. Mostrar Ventana de Resultados
            self.ventana_res = VentanaResultados(
                img_orig    = img_original,
                img_prep    = img_preprocesada,
                mask_celula = mask_celula_completa,
                img_overlay = img_overlay,
                diagnostico = diagnostico,
                relacion_nc = relacion_nc
            )
            self.ventana_res.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Error de Procesamiento", f"Ocurrió un error al analizar la imagen:\n{str(e)}")


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = LeucemiaApp()
    ventana.show()
    sys.exit(app.exec_())
