#vueltas.py
#limpieza de vueltas
def limpiar_vueltas(session, year=None, gp=None, session_type=None):
    laps=session.laps.copy()
    keep_cols = [
        "Driver", "Team", "TeamName",  # (depende de FastF1)
        "LapNumber", "LapTime",
        "Sector1Time", "Sector2Time", "Sector3Time",
        "Compound", "TyreLife", "FreshTyre",
        "PitInTime", "PitOutTime"
    ]
    existing = [c for c in keep_cols if c in laps.columns]
    laps = laps[existing].copy()

    # trazabilidad
    if year is not None: laps["Year"] = year
    if gp is not None: laps["GrandPrix"] = gp
    if session_type is not None: laps["Session"] = session_type

    # quitar pit laps
    if "PitInTime" in laps.columns and "PitOutTime" in laps.columns:
        laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()].copy()

    # quitar LapTime NaN
    laps = laps.dropna(subset=["LapTime"]).copy()

    # LapTime a segundos
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    # outliers: quitar 1% más lento
    q99 = laps["LapTime_s"].quantile(0.99)
    laps = laps[laps["LapTime_s"] <= q99].copy()

    return laps
