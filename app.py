import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Predicción Bienestar Estudiantil", layout="wide")

st.title("Sistema de Predicción de Bienestar Estudiantil")
st.write("Sube un archivo CSV con los datos de nuevos estudiantes.")

# 2. Carga del Modelo
@st.cache_resource
def cargar_modelo():
    try:
        model = joblib.load('modelo_bienestar.joblib')
        return model
    except FileNotFoundError:
        st.error("No se encontró el archivo 'modelo_bienestar.joblib'.")
        return None

rf_model = cargar_modelo()

# 3. Definición de Mapeos (TAL CUAL se usaron en el entrenamiento)
mappings = {
    'Género': {'Femenino': 0, 'Masculino': 1},
    'Horas de sueño': {
        '5-6 horas': 0, 
        '7-8 horas': 1, 
        'menos de 5 horas': 2, 
        'más de 8 horas': 3
    },
    'Hábitos alimenticios': {
        'Moderado': 0, 
        'No Saludable': 1, 
        'Saludable': 2
    },
    'Pensamientos de abandono': {'No': 0, 'Sí': 1}
}

# 4. Función de Clasificación
def clasificar_nivel(prob_pct):
    if prob_pct < 33:
        return "Bajo"
    elif prob_pct < 66:
        return "Medio"
    else:
        return "Alto"

# 5. Interfaz de Carga
uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])

if uploaded_file is not None and rf_model is not None:
    try:
        # Leer archivo
        df_new = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
        
        # Copia para procesar
        X_input = df_new.copy()

        # --- CORRECCIÓN CRÍTICA: Obtener nombres de columnas del modelo ---
        try:
            # Intentamos obtener los nombres de las columnas que el modelo aprendió
            feature_names = rf_model.feature_names_in_
        except AttributeError:
            # Si el modelo es muy antiguo o no guardó los nombres, definimos la lista manualmente
            # (Basado en tu notebook)
            feature_names = [
                'Género', 'Edad', 'Presión académica', 'Satisfacción con estudios',
                'Horas de sueño', 'Hábitos alimenticios', 'Pensamientos de abandono',
                'Horas de estudio', 'Estrés financiero'
            ]
            st.warning("No se pudieron leer los nombres de features del modelo. Usando lista predeterminada.")

        # Verificar que las columnas existan
        missing_cols = set(feature_names) - set(X_input.columns)
        if missing_cols:
            st.error(f"Faltan las siguientes columnas en el archivo subido: {missing_cols}")
            st.stop()

        # 6. Preprocesamiento (Mapeos)
        for col, map_dict in mappings.items():
            if col in X_input.columns:
                # Si es texto, aplicamos map. Si ya es número, lo dejamos.
                if X_input[col].dtype == 'object':
                    X_input[col] = X_input[col].map(map_dict).fillna(0).astype(int)
        
        # 7. Reordenar columnas EXACTAMENTE como el modelo las espera
        X_input = X_input[feature_names]

        # 8. Predicción
        probs = rf_model.predict_proba(X_input)[:, 1]
        
        # 9. Resultados
        df_resultado = df_new.copy()
        df_resultado['Probabilidad_%'] = (probs * 100).round(2)
        df_resultado['Posibilidad'] = df_resultado['Probabilidad_%'].apply(clasificar_nivel)

        st.success("¡Análisis completado con éxito!")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Registros", len(df_resultado))
        col2.metric("Alta Probabilidad", len(df_resultado[df_resultado['Posibilidad'] == 'Alto']))
        col3.metric("Baja Probabilidad", len(df_resultado[df_resultado['Posibilidad'] == 'Bajo']))

        st.dataframe(df_resultado)

        # Descarga
        csv = df_resultado.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            "Descargar Resultados",
            data=csv,
            file_name="predicciones_bienestar.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error detallado: {e}")