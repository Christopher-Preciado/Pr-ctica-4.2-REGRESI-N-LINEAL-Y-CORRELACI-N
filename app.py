import streamlit as st
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración principal de la página
st.set_page_config(
    page_title="Práctica 4.2: Regresión Lineal",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Práctica 4.2: Regresión Lineal y Correlación")
st.subheader("Predicción del Consumo Per Cápita de Agua Potable (CPC)")
st.caption("TecNM Campus Cuautla | Análisis y Exploración de Datos")

# ==============================================================================
# PASO 1: SIMULACIÓN DE DATOS RELACIONALES EN DUCKDB
# ==============================================================================
@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=':memory:')
    con.execute("SELECT setseed(0.42);")
    con.execute("""
    CREATE TABLE telemetria_agua AS
    SELECT
        residencia_id,
        ROUND(18 + (random() * 14), 2) AS temperatura_max,
        ROUND(50 + (random() * 40), 2) AS humedad_max,
        ROUND(3 + (random() * 4), 0) AS personas_familia,
        ROUND(0.2 + (random() * 1.8), 2) AS consumo_bidon_pc,
        ROUND(
            94.0 + (2.5 * (18 + random() * 14))
                 - (0.8 * (50 + random() * 40))
                 - (5.2 * (3 + random() * 4))
                 + (15.0 * (0.2 + random() * 1.8))
                 + (random() * 6.0 - 3.0), 2
        ) AS cpc_consumo
    FROM generate_series(1, 100) AS t(residencia_id);
    """)
    return con.execute("SELECT * FROM telemetria_agua").df()

df_agua = cargar_datos()

# ==============================================================================
# PANEL LATERAL Y VISTA DE DATOS
# ==============================================================================
st.sidebar.header("Opciones de Visualización")
mostrar_df = st.sidebar.checkbox("Mostrar primeros 5 registros de DuckDB", value=True)

if mostrar_df:
    st.markdown("### 1. Primeros Registros Obtenidos de DuckDB")
    st.dataframe(df_agua.head(), use_container_width=True)

# ==============================================================================
# MODALIZACIÓN DE CÁLCULOS ESTADÍSTICOS
# ==============================================================================
matriz_correlacion = df_agua.drop(columns=['residencia_id']).corr(method='pearson')

X = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
Y = df_agua['cpc_consumo']

X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()
residuos = modelo_ols.resid

model_sk = LinearRegression()
model_sk.fit(X, Y)
y_pred = model_sk.predict(X)

r2_sk = r2_score(Y, y_pred)
rmse_sk = np.sqrt(mean_squared_error(Y, y_pred))
stat_sw, p_val_sw = stats.shapiro(residuos)
dw_stat = durbin_watson(residuos)

# ==============================================================================
# MÉTRICAS Y RESUMEN DEL MODELO
# ==============================================================================
st.markdown("---")
st.markdown("### 2. Métricas Clave del Modelo (scikit-learn & statsmodels)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Coeficiente R²", f"{r2_sk * 100:.2f}%")
col2.metric("RMSE", f"{rmse_sk:.4f} lt/hab-día")
col3.metric("Normalidad (p-value)", f"{p_val_sw:.4f}")
col4.metric("Durbin-Watson", f"{dw_stat:.4f}")

with st.expander("Ver Resumen Detallado OLS (statsmodels)"):
    st.text(modelo_ols.summary())

# ==============================================================================
# PANEL GRÁFICO DIAGNÓSTICO
# ==============================================================================
st.markdown("---")
st.markdown("### 3. Panel de Diagnóstico Visual")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Mapa de Calor
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación de Pearson (r)')

# 2. Scatter Plot
sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1],
            line_kws={'color': 'red', 'linewidth': 2})
axes[0, 1].set_title('2. Tendencia: Humedad Máx vs Consumo')
axes[0, 1].set_xlabel('Humedad Máxima (%)')
axes[0, 1].set_ylabel('CPC Consumo (lt/hab-día)')

# 3. Reales vs Predichos
axes[0, 2].scatter(y_pred, Y, alpha=0.7, color='teal')
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)
axes[0, 2].set_title(f'3. Val. Reales vs Predichos (R² = {r2_sk * 100:.1f}%)')
axes[0, 2].set_xlabel('CPC Predicho (ŷ)')
axes[0, 2].set_ylabel('CPC Real (y)')

# 4. Homocedasticidad
axes[1, 0].scatter(y_pred, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('4. Residuos vs Ajustados (Homocedasticidad)')
axes[1, 0].set_xlabel('Valores Predichos (ŷ)')
axes[1, 0].set_ylabel('Residuos (e = y - ŷ)')

# 5. Q-Q Plot
sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal (Residuos)')

axes[1, 2].axis('off')

plt.tight_layout()

# Renderizar en Streamlit
st.pyplot(fig)