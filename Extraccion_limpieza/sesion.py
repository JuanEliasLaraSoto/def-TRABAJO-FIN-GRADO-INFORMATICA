# sesion.py
import fastf1
import os


def _desactivar_verificacion_ssl():
    """
    Algunos entornos (proxy/antivirus interceptando HTTPS) no validan el
    certificado al descargar sesiones aun no cacheadas, con lo que toda
    petición de red falla con SSLCertVerificationError incluso apuntando
    explícitamente al bundle de certifi. fastf1 usa
    requests_cache.CachedSession (subclase de requests.Session), así que
    se parchea Session.request para forzar verify=False. Uso deliberado y
    puntual (verificar_ssl=False en cargar_sesion): los datos descargados
    son públicos (telemetría de FastF1), sin credenciales de por medio.
    """
    import requests
    import urllib3

    if getattr(requests.Session.request, "_ssl_verify_parcheado", False):
        return  # ya parcheado, evita envolver la funcion varias veces

    request_original = requests.Session.request

    def request_sin_verificar(self, *args, **kwargs):
        kwargs["verify"] = False
        return request_original(self, *args, **kwargs)

    request_sin_verificar._ssl_verify_parcheado = True
    requests.Session.request = request_sin_verificar
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


#se activa cache y se carga una sesion(importante realizar el load, poner en memoria)
def cargar_sesion(year, gp, session_type, cache_dir=None, verificar_ssl=True):
    if not verificar_ssl:
        _desactivar_verificacion_ssl()

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
