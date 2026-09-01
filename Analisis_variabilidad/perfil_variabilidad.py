#perfil_variabilidad.py
#carga de telemetria por vuelta y calculo de las metricas de variabilidad
#(una fila por vuelta), mas el cruce con metadatos de vueltas (compuesto,
#vida del neumatico, tiempo de vuelta)
import pandas as pd

from metricas_variabilidad import calcular_todas_metricas

PCT_BAJO = 25
PCT_ALTO = 75


def cargar_telemetria_vueltas(path_interp, path_raw):
    """
    Carga la telemetria interpolada (100Hz) de todas las vueltas limpias del
    piloto de referencia y le añade LapNumber (ausente en el CSV
    interpolado), tomandolo del CSV de telemetria bruta por LapID.
    """
    interp = pd.read_csv(path_interp)
    raw = pd.read_csv(path_raw)
    mapa_lapnumber = raw[["LapID", "LapNumber"]].drop_duplicates().set_index("LapID")["LapNumber"]
    interp["LapNumber"] = interp["LapID"].map(mapa_lapnumber)
    return interp


def calcular_umbrales_fenotipo(df, p_bajo=PCT_BAJO, p_alto=PCT_ALTO):
    """
    Percentiles p_bajo/p_alto de la velocidad agregada de TODAS las vueltas
    de la sesion: umbrales fijos, comunes a todas las vueltas comparadas
    (ver docstring de `metricas_fenotipo` en metricas_variabilidad.py).
    """
    velocidad_agregada = df["Speed"].to_numpy()
    umbral_bajo = float(pd.Series(velocidad_agregada).quantile(p_bajo / 100))
    umbral_alto = float(pd.Series(velocidad_agregada).quantile(p_alto / 100))
    return umbral_bajo, umbral_alto


def calcular_metricas_por_vuelta(df, umbral_bajo, umbral_alto):
    """Aplica `calcular_todas_metricas` a cada vuelta (LapID) por separado."""
    filas = []
    for lap_id, grupo in df.groupby("LapID"):
        grupo = grupo.sort_values("Time_s")
        speed = grupo["Speed"].to_numpy()
        dt = float(grupo["Time_s"].diff().median())
        metricas = calcular_todas_metricas(speed, umbral_bajo, umbral_alto, dt=dt)
        metricas["LapID"] = lap_id
        metricas["LapNumber"] = grupo["LapNumber"].iloc[0]
        filas.append(metricas)
    return pd.DataFrame(filas).sort_values("LapNumber").reset_index(drop=True)


def combinar_con_metadatos(df_metricas, laps_path, driver):
    """Cruza las metricas por vuelta con Compound/TyreLife/LapTime_s de laps_clean.csv."""
    laps = pd.read_csv(laps_path)
    laps_piloto = laps[laps["Driver"] == driver][
        ["LapNumber", "LapTime_s", "Compound", "TyreLife"]
    ]
    return df_metricas.merge(laps_piloto, on="LapNumber", how="left")
