#comparar_frecuencias_interpolacion.py
#funciones para comparar distintas frecuencias de interpolacion (1s, 0.1s,
#0.01s) y justificar la eleccion de 0.01s (100 Hz) en el pipeline, tal y
#como pidio Miguel Angel en la revision del TFG. Orquestado desde main.py
#(--modo comparar-frecuencias).
from telemetria import interpolar_telemetria_temporal

FRECUENCIAS_S = [1.0, 0.1, 0.01]
COLOR_FRECUENCIA = {1.0: "#d62728", 0.1: "#ff7f0e", 0.01: "#1f77b4"}


def resumen_por_frecuencia(df_lap, frecuencia_s, ventana_frenada_s=1.5):
    """Numero de puntos, distancia media entre muestras (a partir de la
    velocidad media) y puntos disponibles dentro de una zona de frenada
    tipica (ventana_frenada_s segundos)."""
    interp = interpolar_telemetria_temporal(df_lap, frecuencia_s=frecuencia_s)
    velocidad_media_ms = (interp["Speed"].mean()) / 3.6
    distancia_media_m = velocidad_media_ms * frecuencia_s
    puntos_en_frenada = max(1, round(ventana_frenada_s / frecuencia_s))
    return {
        "Frecuencia_s": frecuencia_s,
        "Frecuencia_Hz": round(1.0 / frecuencia_s, 2),
        "N_puntos_vuelta": len(interp),
        "Distancia_media_entre_muestras_m": round(distancia_media_m, 2),
        "Puntos_en_frenada_1.5s": puntos_en_frenada,
    }, interp



