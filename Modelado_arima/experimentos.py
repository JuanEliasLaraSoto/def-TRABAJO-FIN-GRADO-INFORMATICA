#experimentos.py
#construye las series agrupadas (pooled) para los dos experimentos de
#generalizacion pedidos por los tutores: division espacial (un trozo del
#circuito para validar, otro distinto para test) y division por vueltas
#(la misma seccion -la vuelta completa-, pero vueltas distintas)
import numpy as np
import pandas as pd

FACTOR_SUBMUESTREO = 20  # 100 Hz -> 5 Hz: el ajuste ARIMA sobre ~400k puntos
                          # (48 vueltas x 8500 puntos) no es viable en tiempo
                          # razonable; se conserva el muestreo regular exigido
                          # por Box-Jenkins, solo con un paso mayor (0.2 s)


def cargar_todas_vueltas_interpoladas(path_interp, path_raw):
    """
    Carga la telemetria interpolada (100 Hz) de todas las vueltas limpias
    y le añade el numero de vuelta real (LapNumber), ausente en el CSV
    interpolado, a partir del CSV de telemetria bruta (mismo LapID).
    """
    interp = pd.read_csv(path_interp)
    raw = pd.read_csv(path_raw)
    mapa_lapnumber = raw[["LapID", "LapNumber"]].drop_duplicates().set_index("LapID")["LapNumber"]
    interp["LapNumber"] = interp["LapID"].map(mapa_lapnumber)
    return interp


def submuestrear_por_vuelta(df, factor=FACTOR_SUBMUESTREO):
    """Toma 1 de cada `factor` puntos dentro de cada vuelta, preservando el muestreo regular."""
    partes = [grupo.sort_values("Time_s").iloc[::factor] for _, grupo in df.groupby("LapID")]
    return pd.concat(partes, ignore_index=True)


def añadir_icc(df, path_curvatura):
    """
    Añade a `df` la columna `ICC`, la variable exogena s(t) del modelo
    ARIMA: el Indice de Complejidad de Curva ya calculado en la fase de
    extraccion (Extraccion_limpieza/telemetria.py, columna `ICC` de
    curvatura_interpolada.csv, equivalente a la curvatura geometrica sin
    signo kappa(s) = |x'y'' - y'x''| / (x'^2+y'^2)^1.5), interpolado en las
    distancias recorridas (`Distance`) de la telemetria. No se recalcula
    nada nuevo: se reutiliza el indice de dificultad ya establecido en el
    pipeline, en lugar de derivar una variante propia para este modulo.

    `Distance` (telemetria) y `s` (curvatura_interpolada.csv) son la misma
    magnitud de arco recorrido desde el inicio de vuelta, por lo que una
    unica interpolacion por distancia sirve para todas las vueltas.
    """
    curvatura = pd.read_csv(path_curvatura)
    s_arco = curvatura["s"].to_numpy()
    icc = curvatura["ICC"].to_numpy()

    df = df.copy()
    df["ICC"] = np.interp(df["Distance"], s_arco, icc)
    return df


def construir_series_experimento_seccion(df, frac_validacion=0.8):
    """
    Experimento 1: para cada vuelta, el primer `frac_validacion` de su
    distancia recorrida (un trozo del circuito) se destina a validacion, y
    el resto (un trozo distinto, el final de cada vuelta) a test. Se agregan
    todas las vueltas disponibles. Devuelve tambien la variable exogena
    s(t) (ICC) alineada punto a punto con cada serie de velocidad.
    """
    val_chunks, test_chunks = [], []
    val_exog_chunks, test_exog_chunks = [], []
    for _, grupo in df.groupby("LapID"):
        grupo = grupo.sort_values("Time_s")
        umbral = frac_validacion * grupo["Distance"].max()
        es_validacion = grupo["Distance"] <= umbral
        val_chunks.append(grupo.loc[es_validacion, "Speed"])
        test_chunks.append(grupo.loc[~es_validacion, "Speed"])
        val_exog_chunks.append(grupo.loc[es_validacion, "ICC"])
        test_exog_chunks.append(grupo.loc[~es_validacion, "ICC"])

    serie_validacion = pd.concat(val_chunks, ignore_index=True)
    serie_test = pd.concat(test_chunks, ignore_index=True)
    exog_validacion = pd.concat(val_exog_chunks, ignore_index=True)
    exog_test = pd.concat(test_exog_chunks, ignore_index=True)
    return serie_validacion, serie_test, exog_validacion, exog_test


def construir_series_experimento_vueltas(df, frac_validacion=0.8):
    """
    Experimento 2: la misma seccion del circuito (la vuelta completa), pero
    vueltas distintas para validacion (las cronologicamente primeras) y
    para test (las cronologicamente ultimas). Devuelve tambien la variable
    exogena s(t) (ICC) alineada punto a punto con cada serie de velocidad.
    """
    vueltas_ordenadas = sorted(df["LapNumber"].dropna().unique())
    n_val = int(len(vueltas_ordenadas) * frac_validacion)
    vueltas_validacion = vueltas_ordenadas[:n_val]
    vueltas_test = vueltas_ordenadas[n_val:]

    df_ordenado = df.sort_values(["LapNumber", "Time_s"])
    es_validacion = df_ordenado["LapNumber"].isin(vueltas_validacion)
    es_test = df_ordenado["LapNumber"].isin(vueltas_test)
    serie_validacion = df_ordenado.loc[es_validacion, "Speed"].reset_index(drop=True)
    serie_test = df_ordenado.loc[es_test, "Speed"].reset_index(drop=True)
    exog_validacion = df_ordenado.loc[es_validacion, "ICC"].reset_index(drop=True)
    exog_test = df_ordenado.loc[es_test, "ICC"].reset_index(drop=True)
    return serie_validacion, serie_test, exog_validacion, exog_test, vueltas_validacion, vueltas_test
