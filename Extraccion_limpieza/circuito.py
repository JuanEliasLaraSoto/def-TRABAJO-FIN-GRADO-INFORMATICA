# calcula longitud aproximada del circuito
def estimar_longitud_circuito(session):
    """
    Estima la longitud del circuito a partir de la distancia acumulada
    ('Distance', en metros) recorrida en la vuelta más rápida de la sesión.

    Es una aproximación: depende de la trazada seguida por el piloto en esa
    vuelta concreta (puede diferir ligeramente de la longitud oficial u
    homologada del circuito) y de la resolución de muestreo de la telemetría
    de FastF1. Se usa como referencia interna del pipeline, no como medida
    oficial del trazado.
    """
    quicklaps = session.laps.pick_quicklaps()
    if quicklaps.empty:
        raise ValueError("No hay 'quicklaps' disponibles en la sesión para estimar la longitud del circuito.")

    lap = quicklaps.pick_fastest()
    tel = lap.get_telemetry()
    return float(tel["Distance"].max())
