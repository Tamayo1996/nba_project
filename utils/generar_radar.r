# generar_radar.R

# Cargar la librería necesaria
suppressPackageStartupMessages(library(fmsb))

# 1. Recibir los argumentos pasados desde Python
args <- commandArgs(trailingOnly = TRUE)
jugador_nombre <- args[1]
triples <- as.numeric(args[2])
ast <- as.numeric(args[3])
reb <- as.numeric(args[4])
robos <- as.numeric(args[5])
tap <- as.numeric(args[6])

# 2. Definir los límites máximos y mínimos de la liga para escalar el gráfico
# (Puedes ajustar estos máximos según los topes históricos de tu dataset)
max_vals <- c(5.0, 10.0, 15.0, 2.5, 3.0)  # Ej: Max 5 triples, 10 ast, 15 reb, etc.
min_vals <- c(0.0, 0.0, 0.0, 0.0, 0.0)
player_vals <- c(triples, ast, reb, robos, tap)

# 3. Construir el dataframe que necesita fmsb (Fila 1: Max, Fila 2: Min, Fila 3: Datos)
data <- as.data.frame(rbind(max_vals, min_vals, player_vals))
colnames(data) <- c("3P", "AST", "TRB", "STL", "BLK")

# 4. Generar y guardar la imagen
png("radar_dinamico.png", width=800, height=800, res=150) # Alta resolución

# Ajustar márgenes
op <- par(mar=c(1, 1, 2, 1))

# Dibujar el radar
radarchart(data, axistype=1, 
           pcol=rgb(0.9, 0.3, 0.2, 0.9),      # Color de la línea (Rojo)
           pfcol=rgb(0.9, 0.3, 0.2, 0.4),     # Color de relleno (Rojo semitransparente)
           plwd=3,                            # Grosor de la línea
           cglcol="grey", cglty=1, cglwd=0.8, # Estilo de la red
           axislabcol="grey",                 # Color de los ejes
           vlcex=1.2,                         # Tamaño de las etiquetas (3P, AST...)
           title=paste("Perfil de Juego:", jugador_nombre))

par(op)
dev.off() # Cerrar y guardar el archivo