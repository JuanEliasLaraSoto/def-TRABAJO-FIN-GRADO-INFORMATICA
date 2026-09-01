#condiciones_meteorologicas.py
#limpieza de las condiciones meteorologicas
def limpiar_condiciones(w):
    """
    Limpieza de las condiciones meteorológicas: elimina filas completamente
    vacías y duplicados exactos, descarta lecturas sin temperatura de aire o
    de pista (canales críticos) y ordena cronológicamente por 'Time' si está
    disponible.
    """
    w = w.dropna(how="all").copy()
    w = w.drop_duplicates().copy()

    critical_cols = [c for c in ["AirTemp", "TrackTemp"] if c in w.columns]
    if critical_cols:
        w = w.dropna(subset=critical_cols).copy()

    if "Time" in w.columns:
        w = w.sort_values("Time").reset_index(drop=True)

    return w

#se obtienen las condiciones meteorologicas de una sesión concreta
def extraer_condiciones(session, year, gp, session_type):
    w = session.weather_data.copy()
    w["Year"] = year
    w["GrandPrix"] = gp
    w["Session"] = session_type
    return w
