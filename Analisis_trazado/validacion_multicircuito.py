#validacion_multicircuito.py
#funciones para leer la geometria reconstruida (curvatura_interpolada.csv)
#de varios circuitos, generada por Extraccion_limpieza/main.py
#(--modo multicircuito), y validar en cuales funciona bien la
#reconstruccion y en cuales no. Orquestado desde main.py (--modo multicircuito).
import os

import pandas as pd

from deteccion_curvas import detectar_curvas, clasificar_dificultad
from mapa_trazado import plot_mapa_curvatura

CIRCUITOS = ["spa", "silverstone", "monaco", "singapur"]


def leer_info_circuito(circuito_dir):
    info = {}
    with open(os.path.join(circuito_dir, "circuit_info.txt"), encoding="utf-8") as f:
        for linea in f:
            clave, _, valor = linea.strip().partition(":")
            info[clave.strip()] = valor.strip()
    return info


def validar_circuito(nombre, data_dir, figures_dir):
    circuito_dir = os.path.join(data_dir, nombre)
    csv_path = os.path.join(circuito_dir, "curvatura_interpolada.csv")
    if not os.path.exists(csv_path):
        print(f"   {nombre}: sin datos extraidos, se omite (ejecuta antes Extraccion_limpieza/main.py --modo multicircuito).")
        return None

    spline_track = pd.read_csv(csv_path)
    info = leer_info_circuito(circuito_dir)

    curvas = detectar_curvas(spline_track, radio_umbral=300.0)
    curvas = clasificar_dificultad(curvas)

    filename = f"mapa_trazado_dificultad_{nombre}.png"
    plot_mapa_curvatura(
        spline_track, figures_dir, filename=filename,
        titulo=f"Mapa de dificultad del trazado — {info.get('Nombre', nombre)}",
    )

    print(f"   {nombre}: {len(curvas)} curvas detectadas, "
          f"error de longitud {info.get('Error longitud (%)', '?')} %")

    return {
        "Circuito": info.get("Nombre", nombre),
        "GP": info.get("GP", ""),
        "Hipotesis": info.get("Hipotesis", ""),
        "Longitud_independiente_m": round(float(info["Longitud independiente (Distance, m)"]), 1),
        "Longitud_spline_m": round(float(info["Longitud reconstruida (spline, m)"]), 1),
        "Error_longitud_pct": round(float(info["Error longitud (%)"]), 2),
        "N_curvas_detectadas": len(curvas),
        "Figura": filename,
    }
