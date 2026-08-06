#heatmaps.py
#mapas de calor: ritmo piloto-vuelta y correlacion entre variables de telemetria
import os

import matplotlib.pyplot as plt
import seaborn as sns


def heatmap_ritmo_piloto_vuelta(laps, output_dir):
    tabla = laps.pivot_table(index="Driver", columns="LapNumber", values="LapTime_s")
    plt.figure(figsize=(14, 8))
    sns.heatmap(tabla, cmap="RdYlGn_r", cbar_kws={"label": "Tiempo de vuelta (s)"})
    plt.xlabel("Número de vuelta")
    plt.ylabel("Piloto")
    plt.title("Mapa de calor del ritmo de carrera (piloto x vuelta)")
    plt.tight_layout()
    path = os.path.join(output_dir, "heatmap_ritmo_piloto_vuelta.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def heatmap_correlacion_telemetria(telemetry, output_dir):
    columnas = [c for c in ["Speed", "Throttle", "RPM", "nGear"] if c in telemetry.columns]
    if len(columnas) < 2:
        return
    corr = telemetry[columnas].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlación entre variables de telemetría")
    plt.tight_layout()
    path = os.path.join(output_dir, "heatmap_correlacion_telemetria.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")
