#cargar_datos.py
import os
import pandas as pd


def cargar_csv(filename, data_dir="../csv_data"):
    path = os.path.join(data_dir, filename)
    return pd.read_csv(path)


def cargar_datos_eda(data_dir="../csv_data"):
    """Carga los artefactos generados por Extraccion_limpieza necesarios para el EDA."""
    laps = cargar_csv("laps_clean.csv", data_dir)
    telemetry_3_mejores = cargar_csv("telemetry_3_mejores.csv", data_dir)
    telemetry_piloto = cargar_csv("telemetry_piloto.csv", data_dir)
    weather = cargar_csv("weather.csv", data_dir)
    return laps, telemetry_3_mejores, telemetry_piloto, weather
