#estadisticas_vueltas.py
#estadisticas y graficas de tiempos de vuelta, sectores y neumaticos
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _sector_cols_a_segundos(laps):
    laps = laps.copy()
    for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[col + "_s"] = pd.to_timedelta(laps[col]).dt.total_seconds()
    return laps


def resumen_por_piloto(laps):
    return (
        laps.groupby("Driver")["LapTime_s"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("median")
        .reset_index()
    )


def resumen_por_equipo(laps):
    if "Team" not in laps.columns:
        return pd.DataFrame()
    return (
        laps.groupby("Team")["LapTime_s"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("median")
        .reset_index()
    )


def plot_boxplot_tiempos_por_piloto(laps, output_dir):
    orden = laps.groupby("Driver")["LapTime_s"].median().sort_values().index
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=laps, x="Driver", y="LapTime_s", order=orden, hue="Driver", legend=False)
    plt.xticks(rotation=45)
    plt.xlabel("Piloto")
    plt.ylabel("Tiempo de vuelta (s)")
    plt.title("Distribución de tiempos de vuelta por piloto")
    plt.tight_layout()
    path = os.path.join(output_dir, "boxplot_tiempos_por_piloto.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_boxplot_tiempos_por_equipo(laps, output_dir):
    if "Team" not in laps.columns:
        return
    orden = laps.groupby("Team")["LapTime_s"].median().sort_values().index
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=laps, x="Team", y="LapTime_s", order=orden, hue="Team", legend=False)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Equipo")
    plt.ylabel("Tiempo de vuelta (s)")
    plt.title("Distribución de tiempos de vuelta por equipo")
    plt.tight_layout()
    path = os.path.join(output_dir, "boxplot_tiempos_por_equipo.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_evolucion_ritmo(laps, output_dir, top_n=6):
    mejores = laps.groupby("Driver")["LapTime_s"].median().sort_values().index[:top_n]
    subset = laps[laps["Driver"].isin(mejores)]
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=subset, x="LapNumber", y="LapTime_s", hue="Driver", marker="o")
    plt.xlabel("Número de vuelta")
    plt.ylabel("Tiempo de vuelta (s)")
    plt.title(f"Evolución del ritmo de carrera (top {top_n} pilotos por mediana)")
    plt.tight_layout()
    path = os.path.join(output_dir, "evolucion_ritmo_carrera.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_efecto_compuesto(laps, output_dir):
    if "Compound" not in laps.columns:
        return
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=laps, x="Compound", y="LapTime_s", hue="Compound", legend=False)
    plt.xlabel("Compuesto de neumático")
    plt.ylabel("Tiempo de vuelta (s)")
    plt.title("Tiempo de vuelta según compuesto de neumático")
    plt.tight_layout()
    path = os.path.join(output_dir, "boxplot_tiempo_por_compuesto.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_efecto_vida_neumatico(laps, output_dir):
    if "TyreLife" not in laps.columns or "Compound" not in laps.columns:
        return
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=laps, x="TyreLife", y="LapTime_s", hue="Compound", alpha=0.6)
    plt.xlabel("Vida del neumático (vueltas)")
    plt.ylabel("Tiempo de vuelta (s)")
    plt.title("Degradación del neumático: tiempo de vuelta vs. vida del neumático")
    plt.tight_layout()
    path = os.path.join(output_dir, "scatter_degradacion_neumatico.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_tiempos_sector(laps, output_dir, top_n=8):
    laps = _sector_cols_a_segundos(laps)
    sector_cols = [c for c in ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"] if c in laps.columns]
    if not sector_cols:
        return
    mejores = laps.groupby("Driver")["LapTime_s"].median().sort_values().index[:top_n]
    subset = laps[laps["Driver"].isin(mejores)]
    melted = subset.melt(
        id_vars="Driver", value_vars=sector_cols, var_name="Sector", value_name="Tiempo_s"
    ).dropna()
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="Driver", y="Tiempo_s", hue="Sector")
    plt.xticks(rotation=45)
    plt.xlabel("Piloto")
    plt.ylabel("Tiempo de sector (s)")
    plt.title(f"Tiempos por sector (top {top_n} pilotos por mediana)")
    plt.tight_layout()
    path = os.path.join(output_dir, "boxplot_tiempos_sector.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")
