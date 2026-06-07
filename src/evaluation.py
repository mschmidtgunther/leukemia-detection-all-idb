import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def evaluar_rendimiento_sistema(y_reales, y_predichas, titulo_grafico="Matriz de Confusión"):
    """
    Evalúa el rendimiento del sistema experto generando una matriz de confusión visual
    y transformando el reporte de clasificación clásico en una tabla ordenada de Pandas.
    
    Args:
        y_reales (list o pd.Series): Etiquetas verdaderas de los frotis.
        y_predichas (list o pd.Series): Etiquetas sugeridas por el sistema experto.
        titulo_grafico (str): Título personalizado para el mapa de calor.
        
    Returns:
        pd.DataFrame: Tabla con las métricas detalladas (Precision, Recall, F1-Score) por clase.
    """
    etiquetas = ['Benign', 'Early', 'Pre', 'Pro']
    
    # 1. GENERACIÓN DE LA MATRIZ DE CONFUSIÓN VISUAL
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_reales, y_predichas, labels=etiquetas)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=etiquetas, 
        yticklabels=etiquetas,
        cbar=True
    )
    plt.title(titulo_grafico, fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Clase Real (Hematólogo)", fontsize=10)
    plt.xlabel("Predicción del Sistema Experto", fontsize=10)
    plt.tight_layout()
    plt.show() # Despliega el gráfico directamente en el notebook
    
    # 2. TRANSFORMACIÓN DEL REPORTE DE TEXTO A TABLA DE PANDAS
    # Usamos output_dict=True para obtener un diccionario estructurado en lugar de un string
    reporte_dict = classification_report(y_reales, y_predichas, output_dict=True)
    
    # Convertimos el diccionario a un DataFrame y lo transponemos para que las clases queden en las filas
    df_metricas = pd.DataFrame(reporte_dict).transpose()
    
    # Redondeamos a 4 decimales para mantener el rigor matemático e institucional
    df_metricas = df_metricas.round(4)
    
    # Renombramos las columnas al español para que luzca óptimo en el reporte escrito
    df_metricas.columns = ['Precisión', 'Exhaustividad (Recall)', 'F1-Score', 'Soporte (Casos)']
    
    # Convertimos la columna de soporte a tipo entero (ya que son conteos de células)
    df_metricas['Soporte (Casos)'] = df_metricas['Soporte (Casos)'].astype(int)
    
    return df_metricas