#metricas_variabilidad.py
#metricas de variabilidad adaptadas de literatura de glucemia (CGM) al perfil
#de velocidad intra-vuelta: cada vuelta hace de "paciente" y su serie
#Speed(t) (a 100Hz) hace de "glucemia a lo largo del dia". Referencias:
#  - Fabris et al. (2019), PLOS ONE 14(5):e0225817 (definiciones de MAGE,
#    CONGA, GFI, TU100/AO140, Poincare SD1/SD2, DFAint/DFAraw)
#  - https://github.com/martinrivero15/cgm-variability-analysis (implementacion
#    de referencia para Poincare, CONGA y GFI)
import numpy as np

FACTOR_SUBMUESTREO = 20  # 100Hz -> 5Hz, igual que Modelado_arima/experimentos.py


def submuestrear(x, factor=FACTOR_SUBMUESTREO):
    """
    Reduce la resolucion de una serie tomando 1 de cada `factor` puntos.

    Las metricas basadas en la transicion punto-a-punto (CONGA, GFI/IFV,
    Poincare) solo tienen sentido fisico si el paso temporal
    entre muestras consecutivas corresponde a una escala de conduccion real
    (frenada/aceleracion), no al micro-suavizado del spline de interpolacion
    a 100Hz. Se reutiliza el mismo factor (20 -> 5Hz) que Modelado_arima usa
    para hacer viable el ajuste ARIMA, por la misma razon de fondo: preservar
    un muestreo regular a un paso mayor.
    """
    x = np.asarray(x, dtype=float)
    return x[::factor]


# ---------------------------------------------------------------------------
# Metricas convencionales
# ---------------------------------------------------------------------------

def metricas_convencionales(x):
    x = np.asarray(x, dtype=float)
    media = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    cv_pct = float(sd / media * 100.0) if media != 0 else float("nan")
    return {"Media_Speed": media, "SD_Speed": sd, "CV_Speed_pct": cv_pct}


# ---------------------------------------------------------------------------
# Metricas de secuencialidad
# ---------------------------------------------------------------------------

def _indices_puntos_giro(x):
    """
    Indices (incluyendo los extremos) de los maximos/minimos locales de `x`,
    es decir, los puntos donde la serie cambia de sentido (sube->baja o
    baja->sube). Es el primer paso del algoritmo clasico de MAGE
    (Service et al., 1970): las excursiones se miden entre puntos de giro
    consecutivos, no entre muestras consecutivas.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return np.arange(n)

    signos = np.sign(np.diff(x))
    # los tramos planos (signo 0) heredan el signo del tramo anterior, para
    # no crear falsos puntos de giro por ruido de resolucion numerica
    for i in range(1, len(signos)):
        if signos[i] == 0:
            signos[i] = signos[i - 1]

    cambios = np.where(np.diff(signos) != 0)[0] + 1  # indice en `x` del giro
    return np.unique(np.concatenate(([0], cambios, [n - 1])))


def mage(x, k=1.0):
    """
    Mean Amplitude of Glycaemic (aqui, de velocidad) Excursion: media de las
    amplitudes entre puntos de giro consecutivos que superan `k` veces la SD
    de toda la vuelta. Con k=1 (umbral usado en el paper de referencia), solo
    sobreviven excursiones grandes (frenadas/curvas completas): el ruido de
    interpolacion queda por debajo del umbral y no contamina la metrica, por
    lo que MAGE puede calcularse a resolucion completa (100Hz).
    """
    x = np.asarray(x, dtype=float)
    sd = np.std(x, ddof=1)
    umbral = k * sd

    giros = _indices_puntos_giro(x)
    if len(giros) < 2:
        return float("nan")

    amplitudes = np.abs(np.diff(x[giros]))
    excursiones = amplitudes[amplitudes > umbral]
    return float(excursiones.mean()) if len(excursiones) > 0 else 0.0


def conga(x, lag=20):
    """
    CONGA(lag): desviacion tipica de las diferencias entre puntos separados
    `lag` muestras. En el paper original, CONGA-2 usa un retardo de 2h sobre
    glucemia muestreada cada 5-15 min; aqui se aplica sobre la serie ya
    submuestreada a 5Hz, por lo que lag=20 equivale a ~4s (una zona de
    frenada/curva tipica), la escala "media" analoga dentro de una vuelta.
    """
    x = np.asarray(x, dtype=float)
    if len(x) <= lag:
        return float("nan")
    diffs = x[lag:] - x[:-lag]
    return float(np.std(diffs, ddof=1))


def gfi(x):
    """
    Indice de Fluctuacion de Velocidad (IFV): adaptacion operativa del
    Glycaemic Fluctuation Index siguiendo la implementacion del proyecto de
    referencia (media del valor absoluto de las diferencias sucesivas), no
    la definicion literal del paper ("area entre el perfil y la media").
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(x))))


# ---------------------------------------------------------------------------
# Fenotipo (adaptacion de TU100 / AO140)
# ---------------------------------------------------------------------------

def metricas_fenotipo(x, umbral_bajo, umbral_alto, dt=0.01):
    """
    Adaptacion de TU100 (tiempo bajo 100 mg/dl) y AO140 (area sobre 140
    mg/dl). En glucemia ambos umbrales son puntos de corte clinicos fijos,
    iguales para todos los pacientes. Aqui no existen umbrales fisiologicos
    universales de velocidad, asi que se usan los percentiles 25/75 de la
    velocidad agregada de TODAS las vueltas limpias de la sesion (fijos para
    todas las vueltas comparadas, igual que 100/140 mg/dl son fijos para
    todos los pacientes) en vez de percentiles propios de cada vuelta, que
    serian tautologicamente ~25%/~75% por construccion y no discriminarian
    nada entre vueltas.

    - TB_p25: % de tiempo de la vuelta con velocidad por debajo del umbral
      bajo (analogo a TU100): cuanto tiempo pasa el piloto en zona lenta.
    - AS_p75: metros de "exceso de velocidad" acumulados por encima del
      umbral alto (analogo a AO140, integrando el exceso en vez de solo
      contar el tiempo, igual que AO140 convierte un umbral binario en una
      variable continua).
    """
    x = np.asarray(x, dtype=float)
    tiempo_bajo_pct = float(np.mean(x < umbral_bajo) * 100.0)
    exceso_kmh = np.clip(x - umbral_alto, 0.0, None)
    area_sobre_m = float(np.sum(exceso_kmh / 3.6) * dt)
    return {"TB_p25_pct": tiempo_bajo_pct, "AS_p75_m": area_sobre_m}


# ---------------------------------------------------------------------------
# Poincare
# ---------------------------------------------------------------------------

def poincare(x):
    """
    SD1: dispersion perpendicular a la diagonal de identidad (variabilidad
    de corto plazo, punto a punto). SD2: dispersion a lo largo de la
    diagonal (variabilidad de largo plazo). E = SD2/SD1 (excentricidad de la
    elipse ajustada), tal como se define en el paper de referencia.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return {"SD1": float("nan"), "SD2": float("nan"), "E_SD2_SD1": float("nan")}
    x1, x2 = x[:-1], x[1:]
    sd1 = float(np.std(x2 - x1, ddof=1) / np.sqrt(2))
    sd2 = float(np.std(x1 + x2, ddof=1) / np.sqrt(2))
    e = sd2 / sd1 if sd1 > 0 else float("nan")
    return {"SD1": sd1, "SD2": sd2, "E_SD2_SD1": e}


# ---------------------------------------------------------------------------
# DFA (Detrended Fluctuation Analysis): DFAint (integrada) y DFAraw (directa)
# ---------------------------------------------------------------------------

def _pendiente_dfa(perfil, min_window=4, max_window=None, n_windows=12):
    n = len(perfil)
    if n < min_window * 4:
        return float("nan")

    ventana_max = max_window or max(min_window + 1, n // 4)
    if ventana_max <= min_window:
        return float("nan")

    tamanos = np.unique(
        np.logspace(np.log10(min_window), np.log10(ventana_max), num=n_windows).astype(int)
    )

    fluctuaciones, validas = [], []
    for w in tamanos:
        if w < 2:
            continue
        n_segmentos = n // w
        if n_segmentos < 2:
            continue
        recortado = perfil[: n_segmentos * w]
        segmentos = recortado.reshape(n_segmentos, w)
        eje_x = np.arange(w)

        rms = []
        for segmento in segmentos:
            coef = np.polyfit(eje_x, segmento, deg=1)
            tendencia = np.polyval(coef, eje_x)
            rms.append(np.sqrt(np.mean((segmento - tendencia) ** 2)))

        f = np.sqrt(np.mean(np.square(rms)))
        if np.isfinite(f) and f > 0:
            validas.append(w)
            fluctuaciones.append(f)

    if len(validas) < 2:
        return float("nan")

    pendiente, _ = np.polyfit(np.log(validas), np.log(fluctuaciones), deg=1)
    return float(pendiente)


def dfa_int(x, **kwargs):
    """DFA clasica: exponente alpha sobre el perfil INTEGRADO (cumsum) de `x`."""
    x = np.asarray(x, dtype=float)
    perfil = np.cumsum(x - x.mean())
    return _pendiente_dfa(perfil, **kwargs)


def dfa_raw(x, **kwargs):
    """
    Variante DFAraw del paper de referencia: se salta el paso de integracion
    y aplica el mismo analisis de fluctuacion sin tendencia directamente
    sobre la serie centrada (sin acumular).
    """
    x = np.asarray(x, dtype=float)
    perfil = x - x.mean()
    return _pendiente_dfa(perfil, **kwargs)


# ---------------------------------------------------------------------------
# Agregado: todas las metricas de una vuelta
# ---------------------------------------------------------------------------

def calcular_todas_metricas(speed, umbral_bajo, umbral_alto, dt=0.01, factor_submuestreo=FACTOR_SUBMUESTREO):
    """
    Calcula el set completo de metricas para el perfil Speed(t) de UNA vuelta.

    - A resolucion completa (100Hz): convencionales, MAGE, DFAint, DFAraw y
      fenotipo (robustas al muestreo fino o directamente definidas sobre el).
    - Submuestreadas a 5Hz: CONGA, IFV (GFI), Poincare
      (metricas de transicion punto-a-punto, ver `submuestrear`).
    """
    speed = np.asarray(speed, dtype=float)
    speed_5hz = submuestrear(speed, factor_submuestreo)

    resultado = {}
    resultado.update(metricas_convencionales(speed))
    resultado["MAGE"] = mage(speed)
    resultado.update(metricas_fenotipo(speed, umbral_bajo, umbral_alto, dt=dt))
    resultado["DFAint_alpha"] = dfa_int(speed)
    resultado["DFAraw_alpha"] = dfa_raw(speed)

    resultado["CONGA"] = conga(speed_5hz)
    resultado["IFV_GFI"] = gfi(speed_5hz)
    resultado.update(poincare(speed_5hz))

    return resultado
