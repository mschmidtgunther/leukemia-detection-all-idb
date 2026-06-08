"Clasificación de las imágenes en base a las características extraídas"

def clasificar_celula_v4(row):
    """
    Sistema Experto V4 basado en reglas biológicas optimizadas por datos.
    Traduce el árbol de decisión puro a condicionales nativos hardcodeados.
    """
    rosado       = row['Porcentaje_Rosado']
    rugosidad    = row['Rugosidad_Nucleo']
    relacion_nc  = row['Relacion_NC']
    area_nucleo  = row['Area_Nucleo']
    
    # --- RAMA PRINCIPAL 1: Células de rugosidad baja/media ---
    if rugosidad <= 25.58:
        if rosado <= 0.80:
            if relacion_nc <= 0.98:
                return 'Pro'
            else:
                return 'Early'
        else: # rosado > 0.80
            if rosado <= 0.93:
                return 'Pro'
            else:
                return 'Benign'
                
    # --- RAMA PRINCIPAL 2: Células muy rugosas (Cromatina muy laxa o alterada) ---
    else: # rugosidad > 25.58
        if rosado <= 0.44:
            # Simplificado: Ambas subramas del árbol daban 'Pre'
            return 'Pre'
        else: # rosado > 0.44
            if rosado <= 0.80:
                return 'Early'
            else:
                return 'Benign'