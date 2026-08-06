# sesion.py
import fastf1
import os

#se activa cache y se carga una sesion(importante realizar el load, poner en memoria)
def cargar_sesion(year, gp, session_type, cache_dir=None):
    if cache_dir is None:
        # Ruta relativa al propio módulo, no al cwd desde el que se ejecute el script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(base_dir, "..", "cache")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, gp, session_type)
    session.load()

    return session
