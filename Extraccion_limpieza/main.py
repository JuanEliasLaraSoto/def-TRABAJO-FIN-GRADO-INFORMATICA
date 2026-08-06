import argparse
import os

import pandas as pd

from sesion import cargar_sesion
from vueltas import limpiar_vueltas
from almacenar_csv import guardar_csv
from telemetria import extraer_telemetria, limpiar_telemetria, interpolar_geometria_y_telemetria_circuito, interpolar_telemetria_temporal#, extraer_tramos_lentos

from condiciones_meteorologicas import extraer_condiciones, limpiar_condiciones
from circuito import estimar_longitud_circuito#, extraer_tramos_lentos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de extracción, limpieza e ingeniería de características de sesiones de F1."
    )
    parser.add_argument("--year", type=int, default=2023, help="Año de la temporada")
    parser.add_argument("--gp", type=str, default="Italian Grand Prix", help="Nombre del Gran Premio")
    parser.add_argument("--session", type=str, default="R", help="Tipo de sesión (R, Q, FP1, FP2, FP3, S...)")
    parser.add_argument("--driver", type=str, default="HAM", help="Piloto de referencia para telemetría individual y geometría del trazado")
    parser.add_argument("--output-dir", type=str, default="../csv_data", help="Directorio de salida para los CSV")
    return parser.parse_args()


args = parse_args()
YEAR = args.year
GP = args.gp
SESSION = args.session
DRIVER = args.driver

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print("🚀 Iniciando pipeline de extracción, limpieza e ingeniería de características...")

# 1) Inicialización y Carga de la Sesión
session = cargar_sesion(YEAR, GP, SESSION)

# 2) Procesamiento y Filtro de Tiempos de Vuelta (vueltas.py)
laps_clean = limpiar_vueltas(session, year=YEAR, gp=GP, session_type=SESSION)
guardar_csv(laps_clean, "laps_clean.csv", output_dir)

# 3) Extracción de Telemetría de Alta Frecuencia (telemetria.py)
telemetry_3_mejores = extraer_telemetria(session, YEAR, GP, SESSION, drivers=None, max_laps_per_driver=3)
telemetry_3_mejores = limpiar_telemetria(telemetry_3_mejores)

telemetry_piloto = extraer_telemetria(session, YEAR, GP, SESSION, drivers=DRIVER, max_laps_per_driver=3)
telemetry_piloto = limpiar_telemetria(telemetry_piloto)

if len(telemetry_3_mejores) > 0:
    guardar_csv(telemetry_3_mejores, "telemetry_3_mejores.csv", output_dir)
else:
    print("⚠️ No se pudo generar telemetry_3_mejores (dataset vacío).")

if len(telemetry_piloto) > 0:
    guardar_csv(telemetry_piloto, "telemetry_piloto.csv", output_dir)
else:
    print("⚠️ No se pudo generar telemetry_piloto (dataset vacío).")

# 4) Condiciones ambientales (condiciones_meteorologicas.py)
weather = extraer_condiciones(session, YEAR, GP, SESSION)
weather = limpiar_condiciones(weather)
if len(weather) > 0:
    guardar_csv(weather, "weather.csv", output_dir)
else:
    print("⚠️ weather_data vacío en esta sesión.")

# 5) Análisis Espacial y Geometría del Circuito (circuito.py)
track_len = estimar_longitud_circuito(session)
# 6) REGULARIZACIÓN DUAL DE TELEMETRÍA Y GEOMETRÍA
if len(telemetry_piloto) > 0:

    # --------------------------------------------------------------------------
    # A) REGULARIZACIÓN TEMPORAL (1D: CubicSpline) -> Para ARIMA / LSTM
    # --------------------------------------------------------------------------
    print("📈 Aplicando interpolación por Splines Cúbicos (1D) a 100 Hz...")
    vueltas_procesadas = []

    for lap_id in telemetry_piloto["LapID"].unique():
        df_lap = telemetry_piloto[telemetry_piloto["LapID"] == lap_id]
        df_lap_interp = interpolar_telemetria_temporal(
            df_lap, frecuencia_s=0.01
        )
        vueltas_procesadas.append(df_lap_interp)

    df_tel_interpolada = pd.concat(vueltas_procesadas, ignore_index=True)
    print(
        f"✅ Telemetría temporal interpolada con éxito. Total filas: {len(df_tel_interpolada)}"
    )

    # Exportación del artefacto interpolado
    df_tel_interpolada.to_csv(
        os.path.join(output_dir, "telemetry_piloto_interpolada.csv"),
        index=False,
    )
    # --------------------------------------------------------------------------
    # B) GEOMETRÍA PARAMÉTRICA 2D (splprep / splev) -> Para Curvatura y Pista
    # --------------------------------------------------------------------------
    print(
        "🗺️ Generando mapa de curvatura paramétrico 2D con splprep / splev..."
    )
    # Seleccionamos la vuelta más rápida de referencia para reconstruir el trazado
    vuelta_rapida_id = (
        laps_clean[laps_clean["Driver"] == DRIVER].sort_values("LapTime_s")
        .iloc[0]["LapNumber"]
    )
    df_lap_reference = telemetry_piloto[
        telemetry_piloto["LapNumber"] == vuelta_rapida_id
    ]

    if df_lap_reference.empty:
        df_lap_reference = telemetry_piloto[
            telemetry_piloto["LapID"] == telemetry_piloto["LapID"].iloc[0]
        ]

    spline_track, meta_track = interpolar_geometria_y_telemetria_circuito(
        df_lap_reference, smoothing=80.0, num_puntos=1500, per=True
    )

    spline_track.to_csv(
        os.path.join(output_dir, "curvatura_interpolada.csv"), index=False
    )
    print(
        f"✅ Geometría 2D reconstruida. Longitud estimada: {meta_track['track_length_m']:.2f} m")

    # --------------------------------------------------------------------------
    # C) SEGMENTACIÓN DE TRAMOS LENTOS (circuito.py)
    # --------------------------------------------------------------------------
    """
    print("📊 Segmentando tramos lentos y chicanes del trazado...")
    df_segments, df_track_curves = extraer_tramos_lentos(
        df_interpolado=spline_track,
        year=YEAR,
        gp=GP,
        session_type=SESSION,
        output_dir=output_dir,
    )
    """

else:
    print(
        "🛑 Error: No se puede proceder a la regularización ni segmentación (telemetría ausente)."
    )
# 8) Almacenamiento y Persistencia de Artefactos de Control (Metadatos)
info_path = os.path.join(output_dir, "circuit_info.txt")
with open(info_path, "w", encoding="utf-8") as f:
    f.write(f"Year: {YEAR}\nGP: {GP}\nSession: {SESSION}\n")
    f.write(f"Estimated track length (m): {track_len}\n")
print(f"🏁 circuit_info.txt se ha guardado con éxito en: {info_path}")
print("🎉 Pipeline finalizado con éxito. Process finished with exit code 0.")
