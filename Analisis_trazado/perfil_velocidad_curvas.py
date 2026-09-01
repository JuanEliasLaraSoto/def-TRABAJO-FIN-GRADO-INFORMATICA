#perfil_velocidad_curvas.py
#cruza la segmentacion geometrica de curvas con la telemetria real del piloto
#(velocidad minima, velocidad de entrada y uso de freno por curva)
import numpy as np
import pandas as pd

VENTANA_ENTRADA_M = 100.0


def _ventana_previa(tel_vuelta, s_inicio, ventana_m, track_length_m):
    """Puntos de telemetría en los `ventana_m` metros previos al inicio de la
    curva, manejando el caso en que esa ventana cruce el cierre de vuelta
    (curva situada justo después de la línea de meta)."""
    s_prev_ini = (s_inicio - ventana_m) % track_length_m
    if s_prev_ini < s_inicio:
        return tel_vuelta[(tel_vuelta["Distance"] >= s_prev_ini) & (tel_vuelta["Distance"] < s_inicio)]
    return tel_vuelta[(tel_vuelta["Distance"] >= s_prev_ini) | (tel_vuelta["Distance"] < s_inicio)]


def calcular_perfil_por_curva(telemetry, curvas, track_length_m, ventana_entrada_m=VENTANA_ENTRADA_M):
    """
    Para cada curva detectada geométricamente (con su rango s_inicio_m/s_fin_m),
    calcula a partir de la telemetría interpolada del piloto (Distance, Speed,
    Brake) la velocidad mínima real de paso, la velocidad de entrada (máxima
    en los `ventana_entrada_m` metros previos) y el porcentaje de tiempo
    frenando dentro del tramo, promediando entre las vueltas disponibles.
    """
    if telemetry.empty or "Distance" not in telemetry.columns:
        return pd.DataFrame()

    lap_ids = telemetry["LapID"].unique()
    resultados = []

    for curva in curvas:
        s_inicio, s_fin = curva["s_inicio_m"], curva["s_fin_m"]
        vels_min, vels_entrada, frenos_pct = [], [], []

        for lap_id in lap_ids:
            tel_vuelta = telemetry[telemetry["LapID"] == lap_id]
            en_curva = tel_vuelta[(tel_vuelta["Distance"] >= s_inicio) & (tel_vuelta["Distance"] <= s_fin)]
            if en_curva.empty:
                continue

            vels_min.append(en_curva["Speed"].min())
            if "Brake" in en_curva.columns:
                frenos_pct.append(float((en_curva["Brake"] > 0.5).mean()) * 100)

            ventana = _ventana_previa(tel_vuelta, s_inicio, ventana_entrada_m, track_length_m)
            if not ventana.empty:
                vels_entrada.append(ventana["Speed"].max())

        if not vels_min:
            continue

        vel_min_media = float(np.mean(vels_min))
        vel_entrada_media = float(np.mean(vels_entrada)) if vels_entrada else None

        resultados.append(
            {
                "Numero": curva["Numero"],
                "N_vueltas": len(vels_min),
                "Velocidad_min_kmh": round(vel_min_media, 1),
                "Velocidad_min_std_kmh": round(float(np.std(vels_min)), 1) if len(vels_min) > 1 else 0.0,
                "Velocidad_entrada_kmh": round(vel_entrada_media, 1) if vel_entrada_media is not None else None,
                "Frenada_delta_kmh": round(vel_entrada_media - vel_min_media, 1) if vel_entrada_media is not None else None,
                "Uso_freno_pct": round(float(np.mean(frenos_pct)), 1) if frenos_pct else None,
            }
        )

    return pd.DataFrame(resultados)
