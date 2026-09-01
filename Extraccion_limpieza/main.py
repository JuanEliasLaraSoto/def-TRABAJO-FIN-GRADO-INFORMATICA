import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

from vueltas import limpiar_vueltas
from almacenar_csv import guardar_csv
from telemetria import extraer_telemetria, limpiar_telemetria, interpolar_geometria_y_telemetria_circuito, interpolar_telemetria_temporal#, extraer_tramos_lentos

from condiciones_meteorologicas import extraer_condiciones, limpiar_condiciones
from circuito import estimar_longitud_circuito#, extraer_tramos_lentos

from comparar_frecuencias_interpolacion import FRECUENCIAS_S, COLOR_FRECUENCIA, resumen_por_frecuencia
from extraer_multicircuito import CIRCUITOS, extraer_circuito

# Rutas fijas: cada modo escribe siempre en el mismo sitio, no hay
# necesidad real de variarlas entre ejecuciones.
OUTPUT_DIR = "../csv_data"
FIGURES_DIR = "../figures/extraccion_limpieza"
TABLES_DIR = "../csv_data/comparar_frecuencias"
MULTICIRCUITO_DIR = "../csv_data/multicircuito"
LAP_ID_REFERENCIA = 1  # vuelta mas rapida (modo comparar-frecuencias)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extracción, limpieza e ingeniería de características de sesiones de F1."
    )
    parser.add_argument(
        "--modo", type=str, default="pipeline",
        choices=["pipeline", "comparar-frecuencias", "multicircuito"],
        help="pipeline: extracción y limpieza estándar (por defecto). "
             "comparar-frecuencias: compara frecuencias de interpolación (1s/0.1s/0.01s). "
             "multicircuito: valida la reconstrucción geométrica en varios circuitos.",
    )
    parser.add_argument("--year", type=int, default=2023, help="Año de la temporada")
    parser.add_argument("--gp", type=str, default="Italian Grand Prix", help="Nombre del Gran Premio")
    parser.add_argument("--session", type=str, default="R", help="Tipo de sesión (R, Q, FP1, FP2, FP3, S...)")
    parser.add_argument("--driver", type=str, default="HAM", help="Piloto de referencia para telemetría individual y geometría del trazado")
    return parser.parse_args()


def ejecutar_pipeline(args):
    # Import diferido: solo este modo necesita fastf1 (via sesion.py), asi
    # otros modos (p. ej. comparar-frecuencias) no lo exigen.
    from sesion import cargar_sesion

    YEAR, GP, SESSION, DRIVER = args.year, args.gp, args.session, args.driver
    output_dir = OUTPUT_DIR
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


    # 8) Almacenamiento y Persistencia de Artefactos de Control (Metadatos)
    info_path = os.path.join(output_dir, "circuit_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Year: {YEAR}\nGP: {GP}\nSession: {SESSION}\n")
        f.write(f"Estimated track length (m): {track_len}\n")
    print(f"🏁 circuit_info.txt se ha guardado con éxito en: {info_path}")

    # 9) EXTRACCIÓN DE TODAS LAS VUELTAS LIMPIAS DEL PILOTO (para Modelado_arima)
    # --------------------------------------------------------------------------
    # telemetry_piloto(.csv/_interpolada.csv) solo contienen las 3 mejores
    # vueltas (suficientes para EDA y Trazado). Los experimentos de
    # generalización de Modelado_arima necesitan particionar por vueltas, por lo
    # que se extraen aquí TODAS las vueltas limpias del piloto, reutilizando las
    # mismas funciones de telemetria.py, en artefactos separados que no alteran
    # telemetry_piloto.csv.
    arima_output_dir = os.path.join(output_dir, "arima")
    os.makedirs(arima_output_dir, exist_ok=True)

    print(f"🏎️ Extrayendo TODAS las vueltas limpias de {DRIVER} (para Modelado_arima)...")
    telemetry_todas_vueltas = extraer_telemetria(session, YEAR, GP, SESSION, drivers=DRIVER, max_laps_per_driver=999)
    telemetry_todas_vueltas = limpiar_telemetria(telemetry_todas_vueltas)
    print(f"   Vueltas extraidas: {telemetry_todas_vueltas['LapID'].nunique()} ({len(telemetry_todas_vueltas)} puntos de telemetría bruta)")
    guardar_csv(telemetry_todas_vueltas, "telemetry_todas_vueltas.csv", arima_output_dir)

    print("📈 Regularizando temporalmente cada vuelta a 100 Hz...")
    vueltas_todas_interp = []
    for lap_id in telemetry_todas_vueltas["LapID"].unique():
        df_lap = telemetry_todas_vueltas[telemetry_todas_vueltas["LapID"] == lap_id]
        vueltas_todas_interp.append(interpolar_telemetria_temporal(df_lap, frecuencia_s=0.01))

    df_todas_interp = pd.concat(vueltas_todas_interp, ignore_index=True)
    df_todas_interp.to_csv(os.path.join(arima_output_dir, "telemetry_todas_vueltas_interpolada.csv"), index=False)
    print(f"   Listo: {df_todas_interp['LapID'].nunique()} vueltas interpoladas, {len(df_todas_interp)} puntos.")

    print("🎉 Pipeline finalizado con éxito. Process finished with exit code 0.")


def ejecutar_comparar_frecuencias(args):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    telemetry = pd.read_csv(os.path.join(OUTPUT_DIR, "telemetry_piloto.csv"))
    df_lap = telemetry[telemetry["LapID"] == LAP_ID_REFERENCIA].sort_values("Time_s")
    print(f"Vuelta de referencia (LapID={LAP_ID_REFERENCIA}): {len(df_lap)} puntos brutos.")

    filas = []
    interpolaciones = {}
    for frecuencia_s in FRECUENCIAS_S:
        fila, interp = resumen_por_frecuencia(df_lap, frecuencia_s)
        filas.append(fila)
        interpolaciones[frecuencia_s] = interp
        print(f"  {frecuencia_s} s ({fila['Frecuencia_Hz']} Hz): {fila['N_puntos_vuelta']} puntos, "
              f"~{fila['Distancia_media_entre_muestras_m']} m entre muestras, "
              f"{fila['Puntos_en_frenada_1.5s']} puntos en una frenada de 1.5 s.")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(os.path.join(TABLES_DIR, "comparativa_frecuencias_interpolacion.csv"), index=False)

    # Zoom en la frenada mas exigente de la sesion de referencia (chicane
    # Rettifilo, Distance ~ 850-1000 m; ver Analisis_trazado)
    ini, fin = 850.0, 1000.0
    raw_zona = df_lap[(df_lap["Distance"] >= ini) & (df_lap["Distance"] <= fin)]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(raw_zona["Distance"], raw_zona["Speed"], color="black", s=18, zorder=5, label="Telemetría bruta (FastF1)")
    for frecuencia_s in FRECUENCIAS_S:
        interp = interpolaciones[frecuencia_s]
        zona = interp[(interp["Distance"] >= ini) & (interp["Distance"] <= fin)]
        ax.plot(
            zona["Distance"], zona["Speed"], marker="o", markersize=3,
            color=COLOR_FRECUENCIA[frecuencia_s], label=f"Interpolado a {frecuencia_s} s ({round(1/frecuencia_s,2)} Hz)",
        )
    ax.set_xlabel("Distancia (m)")
    ax.set_ylabel("Velocidad (km/h)")
    ax.set_title("Comparativa de frecuencias de interpolación en la frenada del Rettifilo")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "comparativa_frecuencias_interpolacion.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")

    print(tabla.to_string(index=False))
    print("Comparativa de frecuencias de interpolacion completada.")


def ejecutar_multicircuito(args):
    os.makedirs(MULTICIRCUITO_DIR, exist_ok=True)

    for circuito in CIRCUITOS:
        try:
            extraer_circuito(circuito, MULTICIRCUITO_DIR)
        except Exception as e:
            print(f"   ERROR en {circuito['nombre']}: {e}")

    print("\nExtracción multicircuito completada.")


if __name__ == "__main__":
    args = parse_args()
    if args.modo == "pipeline":
        ejecutar_pipeline(args)
    elif args.modo == "comparar-frecuencias":
        ejecutar_comparar_frecuencias(args)
    elif args.modo == "multicircuito":
        ejecutar_multicircuito(args)
