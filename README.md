# 🏀 Trabajo Final IA: Predicción de Rendimiento de Jugadores de la NBA

**Asignatura:** Inteligencia Artificial  
**Curso:** 2025–2026  
**Enlace a la aplicación desplegada:** [AÑADIR_AQUÍ_LA_URL_DE_TU_APP_EN_STREAMLIT]

---

## 🎯 1. Objetivo del Proyecto
El objetivo principal de este proyecto es aplicar técnicas de Machine Learning (en concreto, redes neuronales y escalado de datos) para construir una solución aplicada a un problema real: **la estimación del rendimiento futuro de los jugadores de baloncesto de la NBA**. 

Para ello, se ha desarrollado una **aplicación interactiva con Streamlit** que permite a la persona usuaria interactuar con modelos predictivos. El sistema toma como entrada el historial del jugador para proyectar de forma precisa sus métricas o inferir su posición funcional.

## 💡 2. ¿Qué problema resuelve la aplicación?
En el ámbito del análisis deportivo (*Sports Analytics*), la gestión de equipos (franquicias NBA) y las ligas *Fantasy*, estimar cuántos puntos anotará un jugador en la siguiente temporada o analizar su perfil es un desafío complejo debido a la variabilidad del rendimiento. 

Esta aplicación resuelve este problema al ofrecer una interfaz interactiva donde el usuario puede:
- Seleccionar o introducir datos históricos de jugadores.
- Visualizar cómo interactúan variables clave (`PTS_per_game`, `MP_per_game`, `FGA_per_game` y `Age`).
- Ejecutar el modelo predictivo para obtener un resultado observable y cuantificable en tiempo real.
Se pasa de un entorno de predicción puramente teórico a una herramienta interactiva, visual y de fácil acceso para la toma de decisiones.

---

## 🛠️ 3. Reproducibilidad: Entorno, Dependencias y Ejecución

Para garantizar la reproducibilidad mínima del proyecto (tal y como se exige en los requisitos), se utiliza el gestor de paquetes y entornos **`uv`**. Asegúrate de tener instalado `uv` en tu sistema y sigue estos comandos desde la raíz del proyecto:

### Paso 1: Crear el entorno virtual
Crea un entorno aislado especificando la versión de Python (se utiliza la versión 3.11):
```bash
uv venv --python 3.11