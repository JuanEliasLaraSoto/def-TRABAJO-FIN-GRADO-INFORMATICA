import argparse
import os

import pandas as pd

from deteccion_curvas import detectar_curvas, clasificar_dificultad
from mapa_trazado import plot_mapa_curvatura, plot_mapa_curvas_numeradas

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mapa de dificultad del trazado a partir de la curvatura interpolada."
    )
    parser.add_argument("--data-dir", type=str, default="../csv_data")
    parser.add_argument("--figures-dir", type=str, default="../figures")
    parser.add_argument("--tables-dir", type=str, default="../csv_data/eda")
    parser.add_argument(
        "--radio-umbral", type=float, default=300.0,
        help="Radio (m) por debajo del cual un punto se considera parte de una curva",
    )
    return parser.parse_args()


args = parse_args()
os.makedirs(args.figures_dir, exist_ok=True)
os.makedirs(args.tables_dir, exist_ok=True)

print("🗺️ Cargando geometría del trazado...")
spline_track = pd.read_csv(os.path.join(args.data_dir, "curvatura_interpolada.csv"))

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
resumen.to_csv(os.path.join(args.tables_dir, "resumen_curvas_trazado.csv"), index=False)
print(resumen.to_string(index=False))

plot_mapa_curvatura(spline_track, args.figures_dir)
plot_mapa_curvas_numeradas(spline_track, curvas, args.figures_dir, nombres=NOMBRES_MONZA)

print("🎉 Mapa de dificultad del trazado generado con éxito.")
