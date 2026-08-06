#deteccion_curvas.py
#segmentacion de curvas del trazado a partir del radio de curvatura local
import numpy as np

RADIO_UMBRAL_CURVA_M = 300.0


def detectar_curvas(spline_track, radio_umbral=RADIO_UMBRAL_CURVA_M):
    """
    Segmenta el trazado (geometría regularizada, bucle cerrado) en tramos de
    curva contiguos donde Radius_m < radio_umbral, y devuelve una lista con
    un registro por curva (radio mínimo/apex, posición y distancia a la
    que se encuentra).
    """
    es_curva = (spline_track["Radius_m"] < radio_umbral).values
    n = len(es_curva)

    # Rotamos el origen a un punto de recta (si existe) para no partir en
    # dos una curva que cruce el cierre del bucle (s=0 / s=max).
    idx_recta = np.where(~es_curva)[0]
    offset = int(idx_recta[0]) if len(idx_recta) > 0 else 0
    es_curva_rot = np.roll(es_curva, -offset)

    curvas = []
    i = 0
    while i < n:
        if es_curva_rot[i]:
            inicio = i
            while i < n and es_curva_rot[i]:
                i += 1
            fin = i - 1
            idx_original = (np.arange(inicio, fin + 1) + offset) % n
            segmento = spline_track.iloc[idx_original]
            apex_idx = segmento["Radius_m"].idxmin()
            curvas.append(
                {
                    "Apex_idx": apex_idx,
                    "Radio_min_m": float(segmento["Radius_m"].min()),
                    "s_apex_m": float(spline_track.loc[apex_idx, "s"]),
                    "X_apex": float(spline_track.loc[apex_idx, "X"]),
                    "Y_apex": float(spline_track.loc[apex_idx, "Y"]),
                }
            )
        else:
            i += 1

    curvas_ordenadas = sorted(curvas, key=lambda c: c["s_apex_m"])
    for numero, curva in enumerate(curvas_ordenadas, start=1):
        curva["Numero"] = numero

    return curvas_ordenadas


def clasificar_dificultad(curvas):
    """Clasifica cada curva en Baja/Media/Alta según su radio mínimo (terciles)."""
    if not curvas:
        return curvas
    radios = np.array([c["Radio_min_m"] for c in curvas])
    p33, p66 = np.percentile(radios, [33, 66])
    for c in curvas:
        if c["Radio_min_m"] <= p33:
            c["Dificultad"] = "Alta"
        elif c["Radio_min_m"] <= p66:
            c["Dificultad"] = "Media"
        else:
            c["Dificultad"] = "Baja"
    return curvas
