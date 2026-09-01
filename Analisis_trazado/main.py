import argparse
import os

import pandas as pd

from deteccion_curvas import detectar_curvas, clasificar_dificultad
from mapa_trazado import (
    plot_mapa_curvatura,
    plot_mapa_curvas_numeradas,
    plot_relacion_radio_velocidad,
)
from perfil_velocidad_curvas import calcular_perfil_por_curva
from validacion_multicircuito import CIRCUITOS, validar_circuito

# Nombres reales de las curvas del Autodromo Nazionale di Monza, en orden de
# recorrido. Es específico de este circuito (la sesión de referencia del
# proyecto es el GP de Italia): al analizar otro trazado, este diccionario
# no aplica y las curvas se identifican solo por su número.
NOMBRES_MONZA = {
    1: "Rettifilo", 2: "Rettifilo",
    3: "Curva Grande",
    4: "Roggia", 5: "Roggia",
    6: "Lesmo 1",
    7: "Lesmo 2",
    8: "Ascari", 9: "Ascari", 10: "Ascari",
    11: "Parabolica",
}

# Rutas fijas: cada modo escribe siempre en el mismo sitio, no hay
# necesidad real de variarlas entre ejecuciones.
DATA_DIR = "../csv_data"
FIGURES_DIR = "../figures/analisis_trazado"
TABLES_DIR = "../eda/analisis_trazado"
MULTICIRCUITO_DATA_DIR = "../csv_data/multicircuito"
MULTICIRCUITO_FIGURES_DIR = "../figures/analisis_trazado/multicircuito"
MULTICIRCUITO_TABLES_DIR = "../eda/analisis_trazado/multicircuito"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Análisis de la dificultad del trazado a partir de la curvatura interpolada."
    )
    parser.add_argument(
        "--modo", type=str, default="trazado", choices=["trazado", "multicircuito"],
        help="trazado: mapa de dificultad de la sesión de referencia (por defecto). "
             "multicircuito: valida la reconstrucción geométrica en varios circuitos.",
    )
    parser.add_argument(
        "--radio-umbral", type=float, default=300.0,
        help="Radio (m) por debajo del cual un punto se considera parte de una curva",
    )
    return parser.parse_args()


def ejecutar_trazado(args):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("🗺️ Cargando geometría del trazado...")
    spline_track = pd.read_csv(os.path.join(DATA_DIR, "curvatura_interpolada.csv"))

    curvas = detectar_curvas(spline_track, radio_umbral=args.radio_umbral)
    curvas = clasificar_dificultad(curvas)
    print(f"🏁 {len(curvas)} curvas detectadas (umbral de radio: {args.radio_umbral} m).")

    resumen = pd.DataFrame(
        [
            {
                "Numero": c["Numero"],
                "Nombre": NOMBRES_MONZA.get(c["Numero"], ""),
                "Radio_min_m": round(c["Radio_min_m"], 1),
                "Distancia_apex_m": round(c["s_apex_m"], 1),
                "Dificultad": c["Dificultad"],
            }
            for c in curvas
        ]
    )

    telemetry_path = os.path.join(DATA_DIR, "telemetry_piloto_interpolada.csv")
    if os.path.exists(telemetry_path):
        print("🏎️ Cruzando geometría con telemetría real del piloto...")
        telemetry_piloto = pd.read_csv(telemetry_path)
        track_length_m = float(spline_track["s"].max())
        perfil = calcular_perfil_por_curva(telemetry_piloto, curvas, track_length_m)
        if not perfil.empty:
            resumen = resumen.merge(perfil, on="Numero", how="left")
        else:
            print("⚠️ No se pudo calcular el perfil de velocidad por curva (telemetría vacía o sin columna Distance).")
    else:
        print(f"⚠️ No se encontró {telemetry_path}: se omite el cruce con telemetría real.")

    r_radio_velocidad = None
    if "Velocidad_min_kmh" in resumen.columns:
        datos_validos = resumen.dropna(subset=["Radio_min_m", "Velocidad_min_kmh"])
        if len(datos_validos) >= 2:
            r_radio_velocidad = round(float(datos_validos["Radio_min_m"].corr(datos_validos["Velocidad_min_kmh"])), 3)
            resumen["R_Radio_Velocidad"] = r_radio_velocidad

    resumen.to_csv(os.path.join(TABLES_DIR, "resumen_curvas_trazado.csv"), index=False)
    print(resumen.to_string(index=False))

    plot_mapa_curvatura(spline_track, FIGURES_DIR)
    plot_mapa_curvas_numeradas(spline_track, curvas, FIGURES_DIR, nombres=NOMBRES_MONZA)
    if "Velocidad_min_kmh" in resumen.columns:
        plot_relacion_radio_velocidad(resumen, FIGURES_DIR, r=r_radio_velocidad)

    print("🎉 Mapa de dificultad del trazado generado con éxito.")


def ejecutar_multicircuito(args):
    os.makedirs(MULTICIRCUITO_FIGURES_DIR, exist_ok=True)
    os.makedirs(MULTICIRCUITO_TABLES_DIR, exist_ok=True)

    print("Validando reconstruccion geometrica por circuito...")
    resultados = [
        r for r in (
            validar_circuito(nombre, MULTICIRCUITO_DATA_DIR, MULTICIRCUITO_FIGURES_DIR)
            for nombre in CIRCUITOS
        )
        if r is not None
    ]

    df = pd.DataFrame(resultados)
    df.to_csv(os.path.join(MULTICIRCUITO_TABLES_DIR, "validacion_multicircuito.csv"), index=False)
    print("\n" + df.to_string(index=False))
    print("\nValidación multicircuito completada.")


if __name__ == "__main__":
    args = parse_args()
    if args.modo == "trazado":
        ejecutar_trazado(args)
    elif args.modo == "multicircuito":
        ejecutar_multicircuito(args)
