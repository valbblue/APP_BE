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
        # SECCIÓN DEL DASHBOARD
        # ---------------------------------------------------------
        st.divider()
        st.subheader("📊 Dashboard de Resultados")

        # CÁLCULOS PARA KPIs
        total_estudiantes = len(df_resultado)
        
        # Conteos
        count_alto = len(df_resultado[df_resultado['Posibilidad'] == 'Alto'])
        count_medio = len(df_resultado[df_resultado['Posibilidad'] == 'Medio'])
        count_bajo = len(df_resultado[df_resultado['Posibilidad'] == 'Bajo'])
        
        # Porcentajes
        pct_alto = (count_alto / total_estudiantes) * 100 if total_estudiantes > 0 else 0
        pct_medio = (count_medio / total_estudiantes) * 100 if total_estudiantes > 0 else 0
        pct_bajo = (count_bajo / total_estudiantes) * 100 if total_estudiantes > 0 else 0

        # --- FILA 1: KPIs CON PORCENTAJES ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        kpi1.metric(
            label="Total Estudiantes", 
            value=total_estudiantes
        )
        
        kpi2.metric(
            label="Riesgo Alto 🔴", 
            value=count_alto, 
            delta=f"{pct_alto:.1f}% del total",
            delta_color="inverse" # Rojo si aumenta (invertido para riesgo)
        )
        
        kpi3.metric(
            label="Riesgo Medio 🟡", 
            value=count_medio, 
            delta=f"{pct_medio:.1f}% del total",
            delta_color="off" # Gris/Neutro
        )
        
        kpi4.metric(
            label="Riesgo Bajo 🟢", 
            value=count_bajo, 
            delta=f"{pct_bajo:.1f}% del total",
            delta_color="normal" # Verde
        )

        st.markdown("---")

        # --- FILA 2: DISTRIBUCIÓN GENERAL ---
        col_graf1, col_graf2 = st.columns(2)

        with col_graf1:
            # Gráfico de Donas General
            fig_pie = px.pie(
                df_resultado, 
                names='Posibilidad', 
                title='Distribución Total de Riesgos',
                color='Posibilidad',
                color_discrete_map={'Alto':'#FF4B4B', 'Medio':'#FFAA00', 'Bajo':'#00CC96'},
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_graf2:
            # NUEVO: Gráfico Género vs Riesgo
            # Mapeamos 0/1 a texto para que el gráfico se entienda mejor
            df_chart = df_resultado.copy()
            df_chart["Género_Label"] = df_chart["Género"].map({0: "Femenino", 1: "Masculino"})
            
            fig_gender = px.histogram(
                df_chart, 
                x="Género_Label", 
                color="Posibilidad",
                title="Nivel de Riesgo por Género",
                barmode="group", # Barras agrupadas para comparar
                color_discrete_map={'Alto':'#FF4B4B', 'Medio':'#FFAA00', 'Bajo':'#00CC96'},
                category_orders={"Posibilidad": ["Alto", "Medio", "Bajo"]}
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        # --- FILA 3: FOCO EN RIESGO (ALTO + MEDIO) ---
        st.markdown("#### 🚨 Foco: Estudiantes en Alerta (Alto + Medio)")
        
        # Filtramos solo los datos de interés
        df_risk = df_resultado[df_resultado['Posibilidad'].isin(['Alto', 'Medio'])]
        
        if not df_risk.empty:
            col_risk1, col_risk2 = st.columns(2)
            
            with col_risk1:
                # Presión Académica SOLO en estudiantes de riesgo
                # Usamos histograma para ver cuántos hay en cada nivel de presión
                fig_risk_presion = px.histogram(
                    df_risk,
                    x="Presión académica",
                    color="Posibilidad",
                    title="Presión Académica en Estudiantes de Riesgo",
                    color_discrete_map={'Alto':'#FF4B4B', 'Medio':'#FFAA00'},
                    barmode="stack"
                )
                st.plotly_chart(fig_risk_presion, use_container_width=True)
                
            with col_risk2:
                # Estrés Financiero SOLO en estudiantes de riesgo
                fig_risk_finanzas = px.histogram(
                    df_risk,
                    x="Estrés financiero",
                    color="Posibilidad",
                    title="Estrés Financiero en Estudiantes de Riesgo",
                    color_discrete_map={'Alto':'#FF4B4B', 'Medio':'#FFAA00'},
                    barmode="stack"
                )
                st.plotly_chart(fig_risk_finanzas, use_container_width=True)
        else:
            st.info("¡Excelente! No se detectaron estudiantes en Riesgo Alto o Medio para analizar en detalle.")