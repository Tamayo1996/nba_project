# graficos.R
# Carga de librerías esenciales del Tidyverse
library(ggplot2)
library(dplyr)
library(readr)

# 1. Cargar el dataset que ya ha sido procesado por Python
df <- read_csv("df_final.csv", show_col_types = FALSE)

# 2. Gráfico de Barras: Distribución de Posiciones Funcionales
grafico_barras <- df %>%
  filter(Pos %in% c('C', 'PF', 'PG', 'SF', 'SG')) %>%
  ggplot(aes(x = Pos, fill = Pos)) +
  geom_bar(show_legend = FALSE, alpha = 0.85) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Distribución de Posiciones Funcionales en la NBA",
    subtitle = "Análisis de frecuencias sobre el histórico de jugadores",
    x = "Posición Funcional",
    y = "Frecuencia Absoluta"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

# Guardar en alta definición para Streamlit
ggsave("grafico_posiciones.png", plot = grafico_barras, width = 7, height = 4.5, dpi = 300)

# 3. Gráfico de Dispersión: Minutos vs Puntos (Muestreo estadístico)
set.seed(42)
grafico_dispersion <- df %>%
  filter(Pos %in% c('C', 'PF', 'PG', 'SF', 'SG')) %>%
  sample_n(2000) %>%
  ggplot(aes(x = MP_per_game, y = PTS_per_game, color = Pos)) +
  geom_point(alpha = 0.6, size = 1.8) +
  labs(
    title = "Volumen de Puntos vs Minutos en Pista",
    subtitle = "Muestreo aleatorio (n=2000) estratificado por estilo de juego",
    x = "Minutos por Partido",
    y = "Puntos por Partido",
    color = "Posición"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave("grafico_dispersion.png", plot = grafico_dispersion, width = 7, height = 4.5, dpi = 300)

cat("📊 [R SCRIPT] Gráficos de ggplot2 generados y exportados con éxito.\n")