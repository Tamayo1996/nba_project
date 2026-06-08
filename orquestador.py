import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Forzar backend de Keras a PyTorch
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, models

print("🚀 Iniciando el Pipeline Automatizado (Orquestador)...")

# ==========================================
# 1. CARGA DE DATOS MULTIFUENTE (CSV + JSON)
# ==========================================
df = pd.read_csv('df_final.csv')

with open('diccionario_equipos.json', 'r', encoding='utf-8') as f:
    diccionario_equipos = json.load(f)

# ==========================================
# 2. TRANSFORMACIÓN, ORDENACIÓN Y APPLY (Fiel a la filosofía R)
# ==========================================
# Ordenación multicriterio: equivalente al 'arrange' de R (Año ascendente, Puntos descendente)
df.sort_values(by=['Year', 'PTS_per_game'], ascending=[True, False], inplace=True)

# Mapeo plano desde el JSON (Requisito 4 - MAP)
mapeo_nombres = {sigla: datos['nombre'] for sigla, datos in diccionario_equipos.items()}
df['Equipo_Completo'] = df['Tm'].map(mapeo_nombres).fillna("Sin Equipo / Histórico")

# Métrica Avanzada: True Shooting Percentage aproximado mediante un APPLY (Requisito 4 - APPLY)
def calcular_ts_percentage(row):
    puntos = row.get('PTS_per_game', 0)
    tiros_campo = row.get('FGA_per_game', 0)
    tiros_libres = row.get('FTA_per_game', 0)
    denominador = 2 * (tiros_campo + 0.44 * tiros_libres)
    return (puntos / denominador) * 100 if denominador > 0 else 0

df['TS_Percentage'] = df.apply(calcular_ts_percentage, axis=1)
df.to_csv('df_final.csv', index=False)
print("✅ Datos transformados, ordenados (multicriterio) y guardados en 'df_final.csv'.")

# ==========================================
# 3. PREPARACIÓN DE MATRICES Y ESCALADO
# ==========================================
# Limpieza rápida para el entrenamiento
df_clean = df.dropna(subset=['PTS_per_game', 'MP_per_game', 'FGA_per_game', '3PA_per_game', 'FTA_per_game', 'AST_per_game']).copy()

X_pts = df_clean[['MP_per_game', 'FGA_per_game', '3PA_per_game', 'FTA_per_game', 'AST_per_game']].values
y_pts = df_clean['PTS_per_game'].values

X_train, X_test, y_train, y_test = train_test_split(X_pts, y_pts, test_size=0.2, random_state=42)

scaler_pts = StandardScaler()
X_train_scaled = scaler_pts.fit_transform(X_train)
X_test_scaled = scaler_pts.transform(X_test)

with open('scaler_puntos.pkl', 'wb') as f:
    pickle.dump(scaler_pts, f)

# ==========================================
# 4. MODELO BASELINE (Scikit-Learn) vs MODELO AVANZADO (Keras)
# ==========================================
print("\n📋 Entrenando Modelo de Control (Scikit-Learn: Random Forest)...")
model_baseline = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model_baseline.fit(X_train_scaled, y_train)

# Predicciones y Métricas del Baseline
preds_base = model_baseline.predict(X_test_scaled)
mae_base = mean_absolute_error(y_test, preds_base)
rmse_base = np.sqrt(mean_squared_error(y_test, preds_base))

print(f"📊 [SKLEARN BASELINE] MAE: {mae_base:.4f} | RMSE: {rmse_base:.4f}")

print("\n🧠 Entrenando Red Neuronal Avanzada (Keras)...")
inputs = keras.Input(shape=(5,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1, activation='linear')(x)
model_keras = keras.Model(inputs=inputs, outputs=outputs)

model_keras.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_keras.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)

# Predicciones y Métricas de Keras
preds_keras = model_keras.predict(X_test_scaled, verbose=0).flatten()
mae_keras = mean_absolute_error(y_test, preds_keras)
rmse_keras = np.sqrt(mean_squared_error(y_test, preds_keras))

print(f"📊 [KERAS DEEP LEARNING] MAE: {mae_keras:.4f} | RMSE: {rmse_keras:.4f}")

# Guardar el modelo ganador de Keras de manera automática
model_keras.save('modelo_puntos_nba.keras')
print("💾 Modelo 'modelo_puntos_nba.keras' exportado correctamente.")


# ==========================================
# 5. JUSTIFICACIÓN ESTADÍSTICA Y CONEXIÓN CON R
# ==========================================
print("\n⚖️ INFORME DE COMPARATIVA ESTADÍSTICA:")
if mae_keras < mae_base:
    print(f"Ganador: Red Neuronal Keras (Mejora el MAE en {mae_base - mae_keras:.4f} puntos respecto a Sklearn). Justifica el uso de DL.")
else:
    print(f"Ganador: Random Forest Sklearn. El modelo clásico presenta una ventaja competitiva de {mae_keras - mae_base:.4f} puntos.")
print("🏁 ¡Pipeline completado con éxito!")

# Reemplaza desde "import subprocess" hasta el final de tu archivo por esto:
import subprocess

print("\n📊 Conectando ecosistemas: Ejecutando script de R para análisis estadístico...")
try:
    ruta_r = r"C:\Program Files\R\R-4.4.3\bin\x64\Rscript.exe"
    
    # Ejecutamos capturando de forma separada los errores de R
    resultado = subprocess.run([ruta_r, "graficos.R"], check=True, capture_output=True, text=True)
    
    # Si todo va bien, mostramos lo que diga R
    if resultado.stdout:
        print(resultado.stdout)
    print("✅ ¡Pipeline híbrido completado! Los gráficos de R se han integrado correctamente.")

except subprocess.CalledProcessError as e:
    print("\n❌ El script de R se ejecutó, pero falló por un error interno de R:")
    print("-" * 60)
    print(e.stderr)  # <-- Esto te dirá la línea exacta de R que falla
    print("-" * 60)

except Exception as e:
    print(f"\n⚠️ Alerta: No se pudo conectar con R. Error inesperado: {e}")