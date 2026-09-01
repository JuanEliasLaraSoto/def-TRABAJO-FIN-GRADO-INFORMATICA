#extraer_multicircuito.py
#extrae y reconstruye la geometria 2D de varios circuitos distintos, para
#validar la reconstruccion por spline parametrico. Solo usa funciones
#locales de este modulo, igual que main.py. Orquestado desde main.py
#(--modo multicircuito).
import os

from telemetria import extraer_telemetria, limpiar_telemetria, interpolar_geometria_y_telemetria_circuito
from circuito import estimar_longitud_circuito
from almacenar_csv import guardar_csv

# 2 circuitos donde se espera que la reconstruccion funcione bien (trazado
# amplio, curvas bien separadas) y 2 donde se espera que falle o se degrade
# (curvas muy cerradas y/o muy proximas entre si, exigentes para un spline
# con suavizado global fijo)
CIRCUITOS = [
    {"nombre": "spa", "gp": "Belgian Grand Prix", "hipotesis": "bien"},
    {"nombre": "silverstone", "gp": "British Grand Prix", "hipotesis": "bien"},
    {"nombre": "monaco", "gp": "Monaco Grand Prix", "hipotesis": "mal"},
    {"nombre": "singapur", "gp": "Singapore Grand Prix", "hipotesis": "mal"},
]
YEAR = 2023
SESSION = "R"


def extraer_circuito(circuito, output_dir):
    # Import diferido: solo este modo del pipeline necesita fastf1 (via
    # sesion.py), asi otros modos (p. ej. comparar-frecuencias) no lo exigen.
    from sesion import cargar_sesion

    nombre, gp = circuito["nombre"], circuito["gp"]
    print(f"\n=== {nombre} ({gp} {YEAR}) ===")
    circuito_dir = os.path.join(output_dir, nombre)
    os.makedirs(circuito_dir, exist_ok=True)

    # verificar_ssl=False: este entorno concreto no valida el certificado al
    # descargar sesiones aun no cacheadas (ver sesion.py); solo se necesita
    # para estas descargas puntuales de circuitos distintos a Monza.
    session = cargar_sesion(YEAR, gp, SESSION, verificar_ssl=False)
    longitud_independiente = estimar_longitud_circuito(session)

    # una vuelta rapida por piloto: basta para obtener la vuelta mas rapida
    # de la sesion y reconstruir la geometria del trazado a partir de ella
    telemetry = extraer_telemetria(session, YEAR, gp, SESSION, drivers=None, max_laps_per_driver=1)
    telemetry = limpiar_telemetria(telemetry)
    if telemetry.empty:
        print("   Sin telemetria valida, se omite.")
        return None

    mejor_lap_id = telemetry.loc[telemetry["LapTime_s"].idxmin(), "LapID"]
    df_lap_reference = telemetry[telemetry["LapID"] == mejor_lap_id]

    spline_track, meta = interpolar_geometria_y_telemetria_circuito(
        df_lap_reference, smoothing=80.0, num_puntos=1500, per=True
    )
    guardar_csv(spline_track, "curvatura_interpolada.csv", circuito_dir)

    error_pct = 100.0 * abs(meta["track_length_m"] - longitud_independiente) / longitud_independiente
    info_path = os.path.join(circuito_dir, "circuit_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Nombre: {nombre}\nGP: {gp}\nYear: {YEAR}\nSession: {SESSION}\n")
        f.write(f"Hipotesis: {circuito['hipotesis']}\n")
        f.write(f"Longitud independiente (Distance, m): {longitud_independiente}\n")
        f.write(f"Longitud reconstruida (spline, m): {meta['track_length_m']}\n")
        f.write(f"Error longitud (%): {error_pct}\n")

    print(f"   Longitud independiente: {longitud_independiente:.1f} m")
    print(f"   Longitud reconstruida:  {meta['track_length_m']:.1f} m  (error {error_pct:.2f} %)")
    return circuito_dir
