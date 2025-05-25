import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import pydeck as pdk
from sklearn.metrics import precision_score

st.set_page_config(page_title="VentaBoost", layout="wide")

st.markdown("""
    <div style='text-align: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/6/66/Oxxo_Logo.svg' width='200'/>
        <h1 style='margin-top: 10px;'>VentaBoost: Predictor de Éxito de Tiendas</h1>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open('/Users/marielalvarez/Downloads/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('/Users/marielalvarez/Downloads/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('/Users/marielalvarez/Downloads/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

def preprocess(df: pd.DataFrame, encoders: dict, scaler) -> np.ndarray:
    df_proc = df.copy()
    for col, le in encoders.items():
        if col in df_proc.columns:
            df_proc[col] = le.transform(df_proc[col].astype(str))
    drop_cols = [c for c in ['TIENDA_ID', 'exito_70', 'DATASET'] if c in df_proc.columns]
    X = df_proc.drop(columns=drop_cols)
    X_scaled = scaler.transform(X)
    return X_scaled, df_proc

st.markdown("""
    <div style='text-align: center; max-width: 800px; margin: auto;'>
        <p>Sube un archivo <strong>CSV</strong> con atributos de potenciales ubicaciones.<br>
        El sistema predecirá <strong>cuáles tiendas abrir</strong> (1 = abrir, 0 = no abrir),<br>
        indicará <strong>cuántas</strong> y opcionalmente la <strong>exactitud</strong> si tu archivo contiene la columna <code>éxito</code>.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    try:
        X_scaled, df_processed = preprocess(df_raw, encoders, scaler)
    except Exception as e:
        st.error(f"Error al preprocesar: {e}")
        st.stop()

    preds = model.predict(X_scaled)
    df_processed["predicted"] = preds

    selected = df_processed[df_processed["predicted"] == 1]
    n_sel = len(selected)

    st.markdown(f"<div style='text-align: center;'><h3> Se recomiendan <strong>{n_sel}</strong> tiendas para abrir.</h3></div>", unsafe_allow_html=True)

    with st.expander("Ver ID de tiendas recomendadas"):
        st.dataframe(selected[[c for c in ['TIENDA_ID'] if c in selected.columns]])

    if "exito_70" in df_processed.columns:
        precision = precision_score(df_processed["exito_70"], df_processed["predicted"])
        st.metric("Precisión clase 1 (abrir tienda)", f"{precision:.1f}")

    if {'LATITUD_NUM', 'LONGITUD_NUM'}.issubset(selected.columns):
        st.subheader("Ubicación de tiendas exitosas:")
        map_df = selected.rename(columns={'LATITUD_NUM': 'lat', 'LONGITUD_NUM': 'lon'})
        st.map(map_df[['lat', 'lon']])
    else:
        st.info("No se encontraron columnas 'LATITUD_NUM' y 'LONGITUD_NUM' para el mapa.")

    csv_out = df_processed.to_csv(index=False).encode('utf-8')
    st.download_button("⬇Descargar CSV con predicciones", csv_out, file_name="predicciones_tiendas.csv", mime="text/csv")
else:
    st.info("Carga un archivo CSV para comenzar.")
