import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px  # Librería para gráficos bonitos

# 1. Configuración de la página
st.set_page_config(page_title="Predicción Bienestar Estudiantil", layout="wide")

st.title("Sistema de Predicción de Bienestar Estudiantil")
st.write("Sube un archivo CSV para predecir y visualizar los riesgos de deserción.")

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

# 3. Definición de Mapeos
mappings = {
    'Género': {'Femenino': 0, 'Masculino': 1},
    'Horas de sueño': {'5-6 horas': 0, '7-8 horas': 1, 'menos de 5 horas': 2, 'más de 8 horas': 3},
    'Hábitos alimenticios': {'Moderado': 0, 'No Saludable': 1, 'Saludable': 2},
    'Pensamientos de abandono': {'No': 0, 'Sí': 1}
}

# 4. Función de Clasificación
def clasificar_nivel(prob_pct):
    if prob_pct < 33: return "Bajo"
    elif prob_pct < 66: return "Medio"
    else: return "Alto"

# 5. Interfaz de Carga
uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])

if uploaded_file is not None and rf_model is not None:
    try:
        df_new = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
        X_input = df_new.copy()

        # Obtener features del modelo
        try:
            feature_names = rf_model.feature_names_in_
        except AttributeError:
            feature_names = [
                'Género', 'Edad', 'Presión académica', 'Satisfacción con estudios',
                'Horas de sueño', 'Hábitos alimenticios', 'Pensamientos de abandono',
                'Horas de estudio', 'Estrés financiero'
            ]

        # Preprocesamiento
        for col, map_dict in mappings.items():
            if col in X_input.columns:
                if X_input[col].dtype == 'object':
                    X_input[col] = X_input[col].map(map_dict).fillna(0).astype(int)
        
        # Reordenar y Predecir
        X_input = X_input[feature_names]
        probs = rf_model.predict_proba(X_input)[:, 1]
        
        # Resultados
        df_resultado = df_new.copy()
        df_resultado['Probabilidad_%'] = (probs * 100).round(2)
        df_resultado['Posibilidad'] = df_resultado['Probabilidad_%'].apply(clasificar_nivel)

        # ---------------------------------------------------------
        # SECCIÓN DEL DASHBOARD (NUEVO)
        # ---------------------------------------------------------
        st.divider()  # Línea divisoria
        st.subheader("📊 Dashboard de Resultados")

        # Métricas principales (KPIs)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Estudiantes", len(df_resultado))
        kpi2.metric("Riesgo Alto 🔴", len(df_resultado[df_resultado['Posibilidad'] == 'Alto']))
        kpi3.metric("Riesgo Medio 🟡", len(df_resultado[df_resultado['Posibilidad'] == 'Medio']))
        kpi4.metric("Promedio Probabilidad", f"{df_resultado['Probabilidad_%'].mean():.1f}%")

        # Fila 1 de Gráficos
        col_graf1, col_graf2 = st.columns(2)

        with col_graf1:
            # Gráfico de Donas: Distribución de Riesgo
            fig_pie = px.pie(
                df_resultado, 
                names='Posibilidad', 
                title='Distribución de Estudiantes por Nivel de Riesgo',
                color='Posibilidad',
                # Colores semaforo
                color_discrete_map={'Alto':'#FF4B4B', 'Medio':'#FFAA00', 'Bajo':'#00CC96'},
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_graf2:
            # Histograma: Distribución de Probabilidades
            fig_hist = px.histogram(
                df_resultado, 
                x="Probabilidad_%", 
                nbins=20, 
                title="Distribución de Probabilidades (%)",
                color_discrete_sequence=['#3366CC']
            )
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Fila 2 de Gráficos (Análisis de Factores)
        st.markdown("#### 🔍 Análisis de Factores Clave")
        col_graf3, col_graf4 = st.columns(2)

        with col_graf3:
            # Relación Presión Académica vs Riesgo Promedio
            # Agrupamos datos para ver tendencias
            df_presion = df_resultado.groupby("Presión académica")["Probabilidad_%"].mean().reset_index()
            fig_bar = px.bar(
                df_presion, 
                x="Presión académica", 
                y="Probabilidad_%",
                title="Riesgo Promedio según Presión Académica",
                color="Probabilidad_%",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_graf4:
            # Relación Estrés Financiero vs Riesgo Promedio
            df_finanzas = df_resultado.groupby("Estrés financiero")["Probabilidad_%"].mean().reset_index()
            fig_bar2 = px.bar(
                df_finanzas, 
                x="Estrés financiero", 
                y="Probabilidad_%",
                title="Riesgo Promedio según Estrés Financiero",
                color="Probabilidad_%",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_bar2, use_container_width=True)

        # ---------------------------------------------------------
        # TABLA DE DATOS Y DESCARGA
        # ---------------------------------------------------------
        st.divider()
        st.subheader("📋 Tabla de Datos Detallada")
        st.dataframe(df_resultado)

        csv = df_resultado.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            "💾 Descargar Reporte Completo (CSV)",
            data=csv,
            file_name="reporte_bienestar_con_graficos.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error: {e}")