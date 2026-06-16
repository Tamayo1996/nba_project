Trabajo Final IA: Predicción de Rendimiento de Jugadores de la NBA
Asignatura: Inteligencia Artificial

Curso: 2025–2026

Enlace a la aplicación desplegada: https://frasantam.streamlit.app/

1. Objetivo del Proyecto
El objetivo principal de este proyecto es aplicar técnicas de Machine Learning (en concreto, redes neuronales y escalado de datos) para construir una solución aplicada a un problema real: la estimación del rendimiento futuro y la clasificación funcional de los jugadores de baloncesto de la NBA.

Para ello, se ha desarrollado una aplicación interactiva con Streamlit que permite a la persona usuaria interactuar con modelos predictivos. El sistema toma como entrada el historial estadístico del jugador para proyectar de forma precisa sus métricas de anotación o inferir su posición en el campo.

2. ¿Qué problema resuelve la aplicación?
En el ámbito del análisis deportivo, la gestión de equipos (franquicias NBA) y las ligas, estimar el rendimiento futuro de un jugador o analizar su perfil funcional es un desafío complejo debido a la alta variabilidad del juego.

Esta aplicación resuelve este problema al ofrecer una interfaz interactiva donde la persona usuaria puede:

Seleccionar o buscar datos históricos de jugadores reales y comparar sus estadísticas con las predicciones de la IA.

Utilizar un "Laboratorio Ficticio" modificando variables clave mediante controles deslizantes (como Minutos jugados, Tiros intentados, Asistencias, etc.) para observar en tiempo real cómo cambia la predicción del modelo.

Ejecutar proyecciones temporales a futuro basadas en el historial multianual de un jugador.

Con esto, se pasa de un entorno de predicción puramente teórico a una herramienta interactiva, visual y de fácil acceso para la toma de decisiones estratégicas.

3. Estructura del Proyecto
El repositorio está organizado de manera modular para separar los datos, los artefactos de inteligencia artificial y la lógica de la aplicación:

auxiliar_data/: Diccionarios en formato JSON (como el mapeo de franquicias y conferencias) y datasets originales descargados.

data/: Contiene el dataset procesado final (df_final.csv) utilizado por la aplicación web.

models/: Almacena las redes neuronales entrenadas en formato .keras (Clasificación de Posiciones, Predicción de Puntos y Proyección Futura), así como los escaladores matemáticos (.pkl) para el preprocesamiento de datos en tiempo real.

main.py: Script principal que despliega la interfaz web interactiva y gestiona la lógica de usuario en Streamlit.

notebooks/scripts: Archivos de experimentación y entrenamiento (donde se realiza la limpieza de datos y la búsqueda de hiperparámetros con Grid Search).

4. Reproducibilidad: Entorno, Dependencias y Ejecución
Para garantizar la reproducibilidad completa del proyecto siguiendo las mejores prácticas, se utiliza el gestor de paquetes y entornos de alto rendimiento uv.

Sigue estos comandos desde la raíz del proyecto para ejecutar la aplicación en tu máquina local:

Paso 1: Crear el entorno virtual
Crea un entorno aislado especificando la versión de Python requerida para el proyecto (Python 3.12):

Bash
uv venv --python 3.12
Paso 2: Activar el entorno virtual
Activa el entorno dependiendo de tu sistema operativo:

En Linux/macOS:

source .venv/bin/activate

* **En Windows (PowerShell):**
  ```powershell
.venv\Scripts\Activate.ps1
Paso 3: Instalar las dependencias
Instala todas las librerías y dependencias necesarias (Streamlit, Keras, PyTorch, Pandas, Scikit-Learn, etc.) de forma sincronizada y exacta:

Bash
uv sync
Paso 4: Ejecutar la aplicación
Una vez preparado el entorno, lanza el servidor local de Streamlit. Esto abrirá automáticamente la aplicación interactiva en tu navegador web:

Bash
streamlit run main.py
5. Datos Adicionales y Reconstrucción
Si se desea replicar el proceso de entrenamiento desde cero y reconstruir los modelos:

Asegúrate de tener los archivos base (Seasons_Stats.csv, Players.csv, etc.) en la raíz o en auxiliar_data/.

Ejecuta los scripts de preprocesamiento para generar un nuevo data/df_final.csv.

Ejecuta el código de entrenamiento, el cual sobrescribirá automáticamente los artefactos .keras y .pkl en la carpeta models/.