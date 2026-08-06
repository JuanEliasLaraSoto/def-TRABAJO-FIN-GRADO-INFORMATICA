#estadisticas_telemetria.py
#estadisticas y graficas de velocidad, acelerador, freno, marchas y DRS
import os

import matplotlib.pyplot as plt
import seaborn as sns

# En la codificación de canal DRS de FastF1, valores >= 10 (10/12/14)
# indican el alerón móvil realmente desplegado; valores menores solo
# indican que el DRS está disponible/deshabilitado.
DRS_ACTIVO_UMBRAL = 10


def resumen_velocidad_por_piloto(telemetry):
    return (
        telemetry.groupby("Driver")["Speed"]
        .agg(["mean", "max", "std"])
        .sort_values("max", ascending=False)
        .reset_index()
    )


def plot_perfil_velocidad(telemetry, output_dir, max_pilotos=6):
    pilotos = telemetry["Driver"].unique()[:max_pilotos]
    plt.figure(figsize=(12, 6))
    for drv in pilotos:
        datos_piloto = telemetry[telemetry["Driver"] == drv]
        mejor_lap_id = datos_piloto.loc[datos_piloto["LapTime_s"].idxmin(), "LapID"]
        datos_vuelta = datos_piloto[datos_piloto["LapID"] == mejor_lap_id].sort_values("Distance")
        plt.plot(datos_vuelta["Distance"], datos_vuelta["Speed"], label=drv)
    plt.xlabel("Distancia (m)")
    plt.ylabel("Velocidad (km/h)")
    plt.title("Perfil de velocidad en la mejor vuelta de cada piloto")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "perfil_velocidad_distancia.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_distribucion_velocidad(telemetry, output_dir):
    plt.figure(figsize=(8, 6))
    sns.histplot(telemetry["Speed"], bins=50, kde=True)
    plt.xlabel("Velocidad (km/h)")
    plt.title("Distribución de velocidad de todos los puntos de telemetría")
    plt.tight_layout()
    path = os.path.join(output_dir, "histograma_velocidad.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_perfil_acelerador_freno(telemetry_piloto, output_dir):
    if telemetry_piloto.empty:
        return
    lap_id = telemetry_piloto["LapID"].min()
    datos = telemetry_piloto[telemetry_piloto["LapID"] == lap_id].sort_values("Distance")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(datos["Distance"], datos["Throttle"], color="green")
    axes[0].set_ylabel("Acelerador (%)")
    axes[1].plot(datos["Distance"], datos["Brake"].astype(float), color="red")
    axes[1].set_ylabel("Freno (activo=1)")
    axes[1].set_xlabel("Distancia (m)")
    fig.suptitle("Perfil de acelerador y freno a lo largo del trazado")
    plt.tight_layout()
    path = os.path.join(output_dir, "perfil_acelerador_freno.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_velocidad_maxima_por_piloto(telemetry, output_dir):
    resumen = telemetry.groupby("Driver")["Speed"].max().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=resumen.index, y=resumen.values, hue=resumen.index, legend=False)
    plt.xticks(rotation=45)
    plt.xlabel("Piloto")
    plt.ylabel("Velocidad máxima (km/h)")
    plt.title("Velocidad máxima registrada por piloto")
    plt.tight_layout()
    path = os.path.join(output_dir, "barplot_velocidad_maxima.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_uso_drs(telemetry, output_dir):
    if "DRS" not in telemetry.columns:
        return
    telemetry = telemetry.copy()
    telemetry["DRS_activo"] = telemetry["DRS"] >= DRS_ACTIVO_UMBRAL
    resumen = telemetry.groupby("Driver")["DRS_activo"].mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(10, 6))
    sns.barplot(x=resumen.index, y=resumen.values, hue=resumen.index, legend=False)
    plt.xticks(rotation=45)
    plt.xlabel("Piloto")
    plt.ylabel("Puntos de telemetría con DRS activo (%)")
    plt.title("Uso relativo del DRS por piloto")
    plt.tight_layout()
    path = os.path.join(output_dir, "barplot_uso_drs.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_distribucion_marchas(telemetry, output_dir):
    if "nGear" not in telemetry.columns:
        return
    marchas = telemetry["nGear"].dropna().astype(int)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=marchas, hue=marchas, legend=False)
    plt.xlabel("Marcha")
    plt.ylabel("Nº de puntos de telemetría")
    plt.title("Distribución del uso de marchas")
    plt.tight_layout()
    path = os.path.join(output_dir, "distribucion_marchas.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")
