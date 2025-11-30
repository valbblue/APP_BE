import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. Configuración de la página
st.set_page_config(page_title="Predicción Bienestar Estudiantil", layout="wide")

st.title("Sistema de Predicción de Bienestar Estudiantil")
st.markdown("Sube un archivo CSV para generar predicciones y visualizar el análisis de riesgos.")

# 2. Carga del Modelo
@st.cache_resource
def cargar_modelo():
    try:
        model = joblib.load('modelo_bienestar.joblib')
        return model
    except FileNotFoundError:
        st.error("⚠️ Error: No se encontró el archivo 'modelo_bienestar.joblib'.")
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

# 5. Interfaz de Carga y Procesamiento
uploaded_file = st.file_uploader("Cargar archivo CSV (Separador ;)", type=['csv'])

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
        
        # Validar columnas y predecir
        for col in feature_names:
            if col not in X_input.columns:
                X_input[col] = 0
                
        X_input = X_input[feature_names]
        probs = rf_model.predict_proba(X_input)[:, 1]
        
        # Crear DataFrame de Resultados
        df_resultado = df_new.copy()
        df_resultado['Probabilidad_%'] = (probs * 100).round(2)
        df_resultado['Posibilidad'] = df_resultado['Probabilidad_%'].apply(clasificar_nivel)

        # ---------------------------------------------------------
        # DASHBOARD DE RESULTADOS
        # ---------------------------------------------------------
        st.divider()
        st.subheader("📊 Dashboard Analítico de Riesgo")

        # KPIs (Sin flechas, formato limpio)
        total = len(df_resultado)
        count_alto = len(df_resultado[df_resultado["Posibilidad"] == "Alto"])
        count_medio = len(df_resultado[df_resultado["Posibilidad"] == "Medio"])
        count_bajo = len(df_resultado[df_resultado["Posibilidad"] == "Bajo"])
        
        # Cálculo de porcentajes
        pct_alto = (count_alto / total * 100) if total > 0 else 0
        pct_medio = (count_medio / total * 100) if total > 0 else 0
        pct_bajo = (count_bajo / total * 100) if total > 0 else 0

        # Definición de columnas para KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        # Uso de label y value combinados para evitar la flecha (delta)
        kpi1.metric(
            label="Total Estudiantes", 
            value=f"{total}", 
            border=True
        )
        kpi2.metric(
            label="Riesgo Alto 🔴", 
            value=f"{count_alto} ({pct_alto:.1f}%)", 
            border=True
        )
        kpi3.metric(
            label="Riesgo Medio 🟠", 
            value=f"{count_medio} ({pct_medio:.1f}%)", 
            border=True
        )
        kpi4.metric(
            label="Riesgo Bajo 🟢", 
            value=f"{count_bajo} ({pct_bajo:.1f}%)", 
            border=True
        )

        st.markdown("---")

        # --- FILA 1: Resumen General y Análisis de Género ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📌 Distribución General de Riesgo")
            # Gráfico de Donas Atractivo
            fig_pie = px.pie(
                df_resultado, 
                names='Posibilidad', 
                color='Posibilidad',
                color_discrete_map={'Alto':'#FF4B4B', 'Medio':'#FFAA00', 'Bajo':'#00CC96'},
                hole=0.5
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("##### 🚻 Clasificación de Riesgo por Género")
            # Preparar datos: mapear numérico a texto si es necesario para el gráfico
            df_gen = df_resultado.copy()
            if df_gen["Género"].dtype in [int, float, np.int64]:
                 df_gen["Género_label"] = df_gen["Género"].map({0:"Femenino", 1:"Masculino"})
            else:
                 df_gen["Género_label"] = df_gen["Género"]
            
            # Histograma agrupado
            fig_gen = px.histogram(
                df_gen,
                x="Género_label",
                color="Posibilidad",
                barmode="group",
                color_discrete_map={"Alto":"#FF4B4B","Medio":"#FFAA00","Bajo":"#00CC96"},
                text_auto=True
            )
            fig_gen.update_layout(
                xaxis_title=None, 
                yaxis_title="Cantidad de Estudiantes",
                legend_title="Nivel de Riesgo",
                margin=dict(t=30)
            )
            st.plotly_chart(fig_gen, use_container_width=True)

        # --- FILA 2: Factores Críticos (Abandono y Estrés) ---
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("##### 💭 Pensamientos de Abandono vs Nivel de Riesgo")
            # Mapear pensamientos para visualización clara
            df_abandon = df_resultado.copy()
            if df_abandon["Pensamientos de abandono"].dtype in [int, float, np.int64]:
                df_abandon["Abandono_Label"] = df_abandon["Pensamientos de abandono"].map({0: "No", 1: "Sí"})
            else:
                df_abandon["Abandono_Label"] = df_abandon["Pensamientos de abandono"]

            # Gráfico de barras apiladas o agrupadas para ver la relación
            fig_abandon = px.histogram(
                df_abandon,
                x="Abandono_Label",
                color="Posibilidad",
                barmode="relative", # Apilado para ver composición total
                color_discrete_map={"Alto":"#FF4B4B","Medio":"#FFAA00","Bajo":"#00CC96"},
                text_auto=True
            )
            fig_abandon.update_layout(
                xaxis_title="¿Tiene Pensamientos de Abandono?",
                yaxis_title="Cantidad",
                legend_title="Riesgo",
                margin=dict(t=30)
            )
            st.plotly_chart(fig_abandon, use_container_width=True)

        with col4:
            st.markdown("##### 📉 Estrés Académico vs Probabilidad de Deserción")
            # Boxplot para ver distribución de probabilidad por nivel de presión
            # Asumimos que Presión Académica es numérica u ordinal
            fig_stress = px.box(
                df_resultado,
                x="Presión académica",
                y="Probabilidad_%",
                color="Presión académica", # Colorear por nivel de presión para impacto visual
                color_discrete_sequence=px.colors.sequential.Reds,
                points="outliers" # Mostrar solo outliers como puntos para limpieza
            )
            fig_stress.update_layout(
                xaxis_title="Nivel de Presión Académica",
                yaxis_title="Probabilidad Calculada (%)",
                showlegend=False,
                margin=dict(t=30)
            )
            st.plotly_chart(fig_stress, use_container_width=True)

        # ---------------------------------------------------------
        # TABLA Y DESCARGA
        # ---------------------------------------------------------
        st.divider()
        with st.expander("Ver Tabla de Datos Detallada", expanded=False):
            st.dataframe(df_resultado, use_container_width=True)

        csv = df_resultado.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            "💾 Descargar Reporte CSV",
            data=csv,
            file_name="reporte_bienestar_final.csv",
            mime="text/csv",
            type="primary"
        )

    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")