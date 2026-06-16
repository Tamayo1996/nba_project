import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle

# Forzamos a Keras a usar PyTorch como motor de ejecución
os.environ["KERAS_BACKEND"] = "torch"
import keras

st.set_page_config(page_title="NBA AI Predictor", page_icon="🏀", layout="wide")
st.title(" Predicción de Rendimiento de Jugadores de la NBA")

# --- CARGA DE DATOS, MODELOS Y ESCALADORES DESDE SUS NUEVAS CARPETAS ---
@st.cache_resource
def cargar_recursos():
    dir_modelos = 'models'
    ruta_csv = os.path.join('data', 'df_final.csv')
    
    # Modelos
    modelo_pos = keras.models.load_model(os.path.join(dir_modelos, 'modelo_posiciones_nba.keras'))
    modelo_pts = keras.models.load_model(os.path.join(dir_modelos, 'modelo_puntos_nba.keras'))
    modelo_proy = keras.models.load_model(os.path.join(dir_modelos, 'modelo_proyeccion_futuro_nba.keras'))
    
    # Escaladores matemáticos
    with open(os.path.join(dir_modelos, 'scaler_posiciones.pkl'), 'rb') as f:
        sc_pos = pickle.load(f)
    with open(os.path.join(dir_modelos, 'scaler_puntos.pkl'), 'rb') as f:
        sc_pts = pickle.load(f)
    with open(os.path.join(dir_modelos, 'scaler_proyeccion_futuro.pkl'), 'rb') as f:
        sc_proy = pickle.load(f)
        
    # Dataset
    try:
        df = pd.read_csv(ruta_csv)
    except:
        df = None
        
    return modelo_pos, modelo_pts, modelo_proy, sc_pos, sc_pts, sc_proy, df

try:
    model_pos, model_reg, model_proy, scaler_pos, scaler_pts, scaler_proy, df_final = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar recursos (Verifica que las carpetas 'data' y 'models' tengan sus archivos correspondientes): {e}")
    st.stop()

# Calculamos los límites de años de forma dinámica si el dataset existe
if df_final is not None:
    min_year = int(df_final['Year'].min())
    max_year = int(df_final['Year'].max())
else:
    min_year, max_year = 1950, 2024

tab1, tab2, tab3 = st.tabs([
    " Buscador de Jugadores Reales", 
    " Laboratorio de Jugadores Ficticios", 
    " Proyección de Rendimiento Futuro"
])

# ==========================================
# PESTAÑA 1: BUSCADOR REAL
# ==========================================
with tab1:
    st.header("Análisis de Jugadores Históricos (Promedios por Partido)")
    
    col1, col2 = st.columns(2)
    with col1:
        nombre_input = st.text_input("Nombre o Apellido del Jugador:", placeholder="Ej. Curry, Harden, James...", key="nombre_t1")
    with col2:
        ano_input = st.text_input("Año de la Temporada:", placeholder=f"Ej. {max_year}", key="ano_t1")
        st.caption(f" Historial disponible en tu base de datos: desde **{min_year}** hasta **{max_year}**")
        
    if st.button("Analizar Jugador ") and nombre_input and ano_input:
        if df_final is None:
            st.error("No se encuentra el archivo 'df_final.csv' dentro de la carpeta 'data'.")
        else:
            coincidencias = df_final[
                (df_final['Player'].str.contains(nombre_input, case=False, na=False)) & 
                (df_final['Year'].astype(str).str.contains(str(ano_input), na=False))
            ]
            
            if coincidencias.empty:
                st.error(f"No se han encontrado registros para '{nombre_input}' en el año {ano_input}.")
            else:
                jugador = coincidencias.iloc[0]
                st.success(f"¡Jugador encontrado: **{jugador['Player']}** (Temporada {jugador['Year']})!")
                
                mp = float(jugador.get('MP_per_game', 0))
                fga = float(jugador.get('FGA_per_game', 0))
                pa3 = float(jugador.get('3PA_per_game', 0))
                fta = float(jugador.get('FTA_per_game', 0))
                ast = float(jugador.get('AST_per_game', 0))

                triples = float(jugador.get('3P_per_game', 0))
                rebotes = float(jugador.get('TRB_per_game', 0))
                robos = float(jugador.get('STL_per_game', 0))
                tapones = float(jugador.get('BLK_per_game', 0))
                
                puntos_reales = float(jugador.get('PTS_per_game', 0))

                st.write(f" **Estadísticas Reales registradas (Por partido):** Minutos: {mp:.1f} | Tiros Intentados: {fga:.1f} | Triples: {triples:.1f} | Asistencias: {ast:.1f} | Rebotes: {rebotes:.1f}")

                datos_pos = np.array([[triples, ast, rebotes, robos, tapones]])
                datos_reg = np.array([[mp, fga, pa3, fta, ast]])
                
                datos_pos_scaled = scaler_pos.transform(datos_pos)
                datos_reg_scaled = scaler_pts.transform(datos_reg)
                
                try:
                    pred_pos = model_pos.predict(datos_pos_scaled, verbose=0)[0]
                    if np.sum(pred_pos) > 0:
                        pred_pos = pred_pos / np.sum(pred_pos)
                    
                    pred_pts = model_reg.predict(datos_reg_scaled, verbose=0)[0][0]
                except Exception as e:
                    st.error(f"Error en la predicción de la IA: {e}")
                    pred_pos = [0.2] * 5
                    pred_pts = 0.0

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.subheader(" Clasificación de Posición (Modelo 1)")
                    posiciones = ['Pívot (C)', 'Ala-Pívot (PF)', 'Base (PG)', 'Alero (SF)', 'Escolta (SG)']
                    chart_data = pd.DataFrame({'Probabilidad (%)': pred_pos * 100}, index=posiciones)
                    st.bar_chart(chart_data)
                    
                    st.markdown("---")
                    
                    
                with res_col2:
                    st.subheader(" Predicción de Anotación (Modelo 2)")
                    st.metric(label="Puntos por partido PREDICHOS por la IA", value=f"{pred_pts:.2f} PTS")
                    st.metric(label="Puntos por partido REALES anotados", value=f"{puntos_reales:.2f} PTS")
                    error = abs(pred_pts - puntos_reales)
                    st.caption(f"Margen de desviación exacto en este test: {error:.2f} puntos por partido.")
                    st.markdown("---")
                    if os.path.exists("grafico_dispersion.png"):
                        st.image("grafico_dispersion.png", caption="Análisis de desviación muestral (Imagen estática)")

# ==========================================
# PESTAÑA 2: SIMULADOR FICTICIO 
# ==========================================
# ==========================================
# PESTAÑA 2: SIMULADOR FICTICIO 
# ==========================================
with tab2:
    st.header(" Laboratorio de Jugadores Ficticios")
    st.write("Modifica los atributos del jugador. Los bloques diferencian qué estadísticas alimentan a cada modelo.")
    
    sim_col1, sim_col2 = st.columns([1.2, 1.8])
    with sim_col1:
        # --- MOVIDO ARRIBA: Atributos del Modelo 1 ---
        st.markdown("####  Atributos del Estilo de Juego *(Modelo 1)*")
        s_3p = st.slider("Triples anotados (3P):", 0.0, 7.0, 1.0, 0.1)
        s_trb = st.slider("Rebotes totales capturados (TRB):", 0.0, 20.0, 5.0, 0.5)
        s_stl = st.slider("Robos de balón (STL):", 0.0, 5.0, 1.0, 0.1)
        s_blk = st.slider("Tapones colocados (BLK):", 0.0, 5.0, 0.5, 0.1)
        
        st.markdown("---")
        st.markdown("####  Atributo Compartido *(Influye en ambos Modelos)*")
        s_ast = st.slider("Asistencias repartidas (AST):", 0.0, 15.0, 3.0, 0.5)
        
        st.markdown("---")
        # --- MOVIDO ABAJO: Atributos del Modelo 2 ---
        st.markdown("####  Atributos de Anotación *(Modelo 2)*")
        s_mp = st.slider("Minutos en pista (MP):", 5.0, 48.0, 25.0, 0.5)
        s_fga = st.slider("Tiros de campo intentados (FGA):", 1.0, 30.0, 10.0, 0.5)
        s_3pa = st.slider("Triples intentados (3PA):", 0.0, 15.0, 3.0, 0.5)
        s_fta = st.slider("Tiros libres intentados (FTA):", 0.0, 15.0, 3.0, 0.5)
        
    with sim_col2:
        st.subheader(" Análisis del Cerebro Artificial en Tiempo Real")
        
        d_pos_sim = np.array([[s_3p, s_ast, s_trb, s_stl, s_blk]])
        d_reg_sim = np.array([[s_mp, s_fga, s_3pa, s_fta, s_ast]])
        
        d_pos_sim_scaled = scaler_pos.transform(d_pos_sim)
        d_reg_sim_scaled = scaler_pts.transform(d_reg_sim)
        
        try:
            p_pos_sim = model_pos.predict(d_pos_sim_scaled, verbose=0)[0]
            if np.sum(p_pos_sim) > 0:
                p_pos_sim = p_pos_sim / np.sum(p_pos_sim)
                
            p_pts_sim = model_reg.predict(d_reg_sim_scaled, verbose=0)[0][0]
        except:
            p_pos_sim = [0.2] * 5
            p_pts_sim = 0.0
            
            
        pos_labels = ['Pívot (C)', 'Ala-Pívot (PF)', 'Base (PG)', 'Alero (SF)', 'Escolta (SG)']
        sim_chart_data = pd.DataFrame({'Probabilidad (%)': p_pos_sim * 100}, index=pos_labels)
        st.bar_chart(sim_chart_data)
        
        st.markdown("---")
        st.metric(label=" Proyección de Puntos Estimados por la IA", value=f"{p_pts_sim:.1f} Puntos por Partido")

# ==========================================
# PESTAÑA 3: PROYECCIÓN A FUTURO
# ==========================================
with tab3:
    st.header(" Proyección Temporal a Futuro (Modelo Adaptativo Multianual)")
    st.write("Introduce un jugador y el año que quieres predecir.")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        nombre_proy = st.text_input("Nombre o Apellido del Jugador a proyectar:", placeholder="Ej. LeBron, Jordan, Curry...", key="nombre_p")
    with p_col2:
        ano_proy = st.text_input("Año que deseas PREDECIR:", placeholder=f"Ej. {max_year}")
        st.caption(f" Recuerda: El rango de tu base de datos histórica es {min_year} - {max_year}")

    if st.button("Calcular Proyección Futura ") and nombre_proy and ano_proy:
        if df_final is None:
            st.error("No se encuentra el archivo 'df_final.csv' dentro de la carpeta 'data'.")
        else:
            try:
                target_year = int(ano_proy)
            except:
                st.error("Por favor, introduce un año numérico válido.")
                st.stop()

            historial = df_final[df_final['Player'].str.contains(nombre_proy, case=False, na=False)].sort_values('Year')

            if historial.empty:
                st.error(f"No se han encontrado registros para el jugador '{nombre_proy}'.")
            else:
                anos_reales_datos = []
                for k in range(1, 6):
                    ano_buscado = target_year - k
                    fila_ano = historial[historial['Year'] == ano_buscado]
                    if not fila_ano.empty:
                        anos_reales_datos.insert(0, fila_ano.iloc[0])
                    else:
                        break

                if len(anos_reales_datos) == 0:
                    st.error(f"Faltan datos de la temporada anterior ({target_year-1}) para proyectar el año {target_year}.")
                else:
                    num_anos_reales = len(anos_reales_datos)
                    num_anos_faltantes = 5 - num_anos_reales

                    fila_historial = []
                    for _ in range(num_anos_faltantes):
                        fila_historial.extend([0.0, 0.0, 0.0, 0.0])

                    anos_detectados_str = []
                    for registro in anos_reales_datos:
                        pts_g = float(registro.get('PTS_per_game', 0))
                        mp_g = float(registro.get('MP_per_game', 0))
                        fga_g = float(registro.get('FGA_per_game', 0))
                        edad = float(registro.get('Age', 0))
                        
                        fila_historial.extend([pts_g, mp_g, fga_g, edad])
                        anos_detectados_str.append(f"{int(registro['Year'])}")

                    datos_proy_vector = np.array([fila_historial])
                    datos_proy_scaled = scaler_proy.transform(datos_proy_vector)

                    try:
                        pred_futura = model_proy.predict(datos_proy_scaled, verbose=0)[0][0]
                    except Exception as e:
                        st.error(f"Error en el modelo de futuro: {e}")
                        pred_futura = 0.0

                    st.info(f" **Historial consecutivo analizado:** {', '.join(anos_detectados_str)}")

                    fila_futuro_real = historial[historial['Year'] == target_year]

                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.subheader(f" Proyección para {target_year}")
                        st.metric(label="Puntos por partido PREDICHOS", value=f"{pred_futura:.2f} PTS")
                        
                    with col_res2:
                        st.subheader(f" Rendimiento Real en {target_year}")
                        if not fila_futuro_real.empty:
                            pts_fut_real = float(fila_futuro_real.iloc[0].get('PTS_per_game', 0))
                            st.metric(label="Puntos por partido REALES", value=f"{pts_fut_real:.2f} PTS")
                            error_proy = abs(pred_futura - pts_fut_real)
                            st.caption(f"Desviación de la proyección: {error_proy:.2f} puntos.")
                        else:
                            st.warning(f"No hay registros de {target_year} en el CSV. ¡Predicción a futuro real!")