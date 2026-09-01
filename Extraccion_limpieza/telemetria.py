import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, splev, splprep

def limpiar_telemetria(tel):
    # mínimo: asegurar que hay velocidad
    if "Speed" in tel.columns:
        tel = tel.dropna(subset=["Speed"]).copy()
        tel = tel[tel["Speed"] >= 0].copy()
    return tel

def extraer_telemetria(session, year, gp, session_type, drivers=None, max_laps_per_driver=3):
    """
    Extrae la telemetría de las mejores vueltas de la sesión.

    Si `drivers` es None, se procesan las mejores vueltas de todos los
    pilotos. Si `drivers` es un string o una lista de strings, se restringe
    la extracción a esos pilotos (p. ej. drivers="HAM" o drivers=["HAM", "VER"]).
    """
    laps = session.laps.pick_quicklaps()

    if drivers is not None:
        if isinstance(drivers, str):
            drivers = [drivers]
        laps = laps[laps["Driver"].isin(drivers)]
        if laps.empty:
            print(f"⚠️ Alerta: ninguno de los pilotos {drivers} tiene vueltas válidas en {year} {gp} ({session_type}).")
            return pd.DataFrame()

    rows = []
    lap_id = 1
    for drv in laps["Driver"].unique():
        best = laps[laps["Driver"] == drv].sort_values("LapTime").head(max_laps_per_driver)
        for _, lap in best.iterrows():
            try:
                tel = lap.get_telemetry()

                if tel.empty:
                    continue

                #Evita errores si alguna columna no existe.
                keep = [c for c in ["Time","Distance","X","Y","Speed","Throttle","Brake","RPM","nGear","DRS"] if c in tel.columns]
                tel = tel[keep].copy()
                tel["LapID"] = lap_id
                tel["PointIndex"] = range(len(tel))
                tel["Year"] = year
                tel["GrandPrix"] = gp
                tel["Session"] = session_type
                tel["Driver"] = drv
                tel["LapNumber"] = lap["LapNumber"]
                tel["LapTime_s"] = lap["LapTime"].total_seconds()
                tel["Time_s"] = tel["Time"].dt.total_seconds()
                lap_id += 1
                rows.append(tel)
            except Exception as e:
                print(f"⚠️ Error al extraer telemetría en la vuelta {lap['LapNumber']} de {drv}: {e}")
                continue

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def interpolar_telemetria_temporal(
    df_lap: pd.DataFrame, frecuencia_s: float = 0.1
) -> pd.DataFrame:

    """
    Realiza una regularización temporal explícita (1D) vuelta por vuelta mediante

    CubicSpline para sincronizar la telemetría a un paso constante.

    :param df_lap: DataFrame con la telemetría de una única vuelta.
    :param frecuencia_s: Paso temporal deseado en segundos (default 0.1s = 10Hz).
    :return: DataFrame regularizado a frecuencia constante.
    """

    columnas_criticas = ["Time_s", "Speed", "LapID"]
    if not all(col in df_lap.columns for col in columnas_criticas):
        return df_lap

    # 1. Ordenar y eliminar duplicados en Time_s (requisito estricto de CubicSpline)
    df_vuelta = (
        df_lap.sort_values("Time_s")
        .drop_duplicates(subset=["Time_s"])
        .copy()
    )

    # CubicSpline (bc_type='not-a-knot' por defecto) necesita al menos 4 puntos
    if len(df_vuelta) < 4:
        print(f"⚠️ Vuelta con solo {len(df_vuelta)} puntos válidos: no se puede interpolar, se devuelve sin regularizar.")
        return df_lap

    # 2. Generar la nueva cuadrícula temporal constante
    tiempo_min = df_vuelta["Time_s"].min()
    tiempo_max = df_vuelta["Time_s"].max()
    tiempo_regular = np.arange(tiempo_min, tiempo_max, frecuencia_s)

    if len(tiempo_regular) == 0:
        print("⚠️ Rango temporal insuficiente para la frecuencia solicitada: se devuelve sin regularizar.")
        return df_lap

    df_interpolado = pd.DataFrame({"Time_s": tiempo_regular})
    df_interpolado["LapID"] = df_vuelta["LapID"].iloc[0]

    # 3. Interpolación 1D con CubicSpline para variables temporales monótonas
    cs_speed = CubicSpline(
        df_vuelta["Time_s"].values, df_vuelta["Speed"].values
    )
    df_interpolado["Speed"] = cs_speed(tiempo_regular)

    if "Distance" in df_vuelta.columns:
        cs_dist = CubicSpline(
            df_vuelta["Time_s"].values, df_vuelta["Distance"].values
        )
        df_interpolado["Distance"] = cs_dist(tiempo_regular)

    # Opcional: si existen canales de pedales los interpolamos igual
    for canal in ["Throttle", "Brake"]:
        if canal in df_vuelta.columns:
            cs_canal = CubicSpline(
                df_vuelta["Time_s"].values, df_vuelta[canal].values
            )
            df_interpolado[canal] = cs_canal(tiempo_regular)

    return df_interpolado


def interpolar_geometria_y_telemetria_circuito(
    df_tel: pd.DataFrame,
    smoothing: float = 80.0,
    num_puntos: int = 1500,
    per: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Reconstruye la geometría 2D del circuito mediante Splines Cúbicos Paramétricos (splprep/splev)

    y mapea síncronamente los canales cinemáticos (Speed, Throttle, Brake) sobre
    el eje de progreso u in [0, 1].
    """
    # Aseguramos que la telemetría esté ordenada espacialmente
    df_tel = df_tel.sort_values(by="Distance").copy()

    # FastF1 expresa X/Y en decimetros (1/10 m), mientras que el resto del
    # pipeline (p. ej. Distance, usada en estimar_longitud_circuito) trabaja
    # en metros. Sin este reescalado, la longitud de arco, la curvatura y el
    # radio resultantes quedan distorsionados por un factor de 10x.
    eje_x = df_tel["X"].values / 10.0
    eje_y = df_tel["Y"].values / 10.0

    # 1. Ajuste paramétrico de la curva plana 2D para la Pista (X(u), Y(u))
    tck, u_original = splprep([eje_x, eje_y], s=float(smoothing), per=per, k=3)

    # 2. Creación del nuevo dominio paramétrico homogéneo u en [0, 1]
    u_new = np.linspace(0.0, 1.0, num_puntos)

    # 3. Evaluación de las coordenadas geoespaciales y derivadas analíticas
    x_curva, y_curva = splev(u_new, tck)
    dx, dy = splev(u_new, tck, der=1)
    ddx, ddy = splev(u_new, tck, der=2)

    x_arr, y_arr = np.array(x_curva, dtype=float), np.array(
        y_curva, dtype=float
    )
    dx_arr, dy_arr = np.array(dx, dtype=float), np.array(dy, dtype=float)
    ddx_arr, ddy_arr = np.array(ddx, dtype=float), np.array(ddy, dtype=float)

    # 4. Distancia acumulada sobre la pista suavizada
    ds = np.hypot(np.diff(x_arr), np.diff(y_arr))
    s_cum = np.r_[0.0, np.cumsum(ds)]

    # 5. Cálculo analítico de la Curvatura kappa
    denom = np.power(dx_arr**2 + dy_arr**2, 1.5)
    denom = np.where(denom == 0, np.nan, denom)
    curvature = np.abs(dx_arr * ddy_arr - dy_arr * ddx_arr) / denom



    # 7. Consolidación de la matriz unificada
    data_dict = {
        "u": u_new,
        "s": s_cum,
        "X": x_arr,
        "Y": y_arr,
        "dx": dx_arr,
        "dy": dy_arr,
        "ddx": ddx_arr,
        "ddy": ddy_arr,
        "Curvature": curvature,
        "ICC": curvature,  # Índice de Complejidad de Curva
        "Radius_m": 1.0 / np.where(curvature == 0, np.nan, curvature),
    }

    # Unimos los canales cinemáticos interpolados

    spline_track = pd.DataFrame(data_dict)

    meta = {
        "raw_points": int(len(df_tel)),
        "spline_points": int(len(spline_track)),
        "track_length_m": float(spline_track["s"].max()),
    }

    return spline_track, meta


