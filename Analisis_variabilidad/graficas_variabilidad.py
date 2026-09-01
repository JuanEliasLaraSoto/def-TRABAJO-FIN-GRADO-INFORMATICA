#graficas_variabilidad.py
#graficas para las metricas de variabilidad por vuelta
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

METRICAS_DISTRIBUCION = [
    "CV_Speed_pct", "MAGE", "CONGA", "IFV_GFI",
    "DFAint_alpha", "DFAraw_alpha",
]

ETIQUETAS = {
    "CV_Speed_pct": "CV velocidad (%)",
    "MAGE": "MAGE (km/h)",
    "CONGA": "CONGA (km/h)",
    "IFV_GFI": "IFV / GFI (km/h)",
    "DFAint_alpha": r"DFA$_{int}$ ($\alpha$)",
    "DFAraw_alpha": r"DFA$_{raw}$ ($\alpha$)",
    "TB_p25_pct": "Tiempo bajo p25 (%)",
    "AS_p75_m": "Área sobre p75 (m)",
    "SD1": "Poincaré SD1 (km/h)",
    "SD2": "Poincaré SD2 (km/h)",
    "LapTime_s": "Tiempo de vuelta (s)",
    "TyreLife": "Vida del neumático (vueltas)",
}


def _etiqueta(col):
    return ETIQUETAS.get(col, col)


def plot_distribuciones_metricas(df, output_dir, metricas=METRICAS_DISTRIBUCION):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metrica in zip(axes.flat, metricas):
        sns.histplot(df[metrica].dropna(), bins=12, kde=True, ax=ax)
        ax.set_xlabel(_etiqueta(metrica))
        ax.set_ylabel("Nº de vueltas")
    fig.suptitle("Distribución de métricas de variabilidad de velocidad (una vuelta = un caso)")
    plt.tight_layout()
    path = os.path.join(output_dir, "distribuciones_metricas_variabilidad.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_metrica_vs_ritmo(df, metrica, output_dir):
    datos = df.dropna(subset=[metrica, "LapTime_s"])
    if len(datos) < 2:
        return None
    r = float(datos[metrica].corr(datos["LapTime_s"]))

    plt.figure(figsize=(8, 6))
    sns.regplot(x=metrica, y="LapTime_s", data=datos, scatter_kws={"s": 40})
    plt.xlabel(_etiqueta(metrica))
    plt.ylabel(_etiqueta("LapTime_s"))
    plt.title(f"{_etiqueta(metrica)} vs. tiempo de vuelta (r={r:.3f})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{metrica.lower()}_vs_ritmo.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")
    return r


def plot_metrica_vs_tyrelife(df, metrica, output_dir):
    datos = df.dropna(subset=[metrica, "TyreLife"])
    if len(datos) < 2:
        return None
    r = float(datos[metrica].corr(datos["TyreLife"]))

    plt.figure(figsize=(8, 6))
    sns.regplot(x="TyreLife", y=metrica, data=datos, scatter_kws={"s": 40})
    plt.xlabel(_etiqueta("TyreLife"))
    plt.ylabel(_etiqueta(metrica))
    plt.title(f"{_etiqueta(metrica)} vs. vida del neumático (r={r:.3f})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{metrica.lower()}_vs_tyrelife.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")
    return r


def plot_boxplot_por_compuesto(df, metrica, output_dir):
    if "Compound" not in df.columns or df["Compound"].nunique() < 2:
        return
    plt.figure(figsize=(7, 6))
    sns.boxplot(x="Compound", y=metrica, data=df, hue="Compound", legend=False)
    sns.stripplot(x="Compound", y=metrica, data=df, color="black", alpha=0.5, size=4)
    plt.xlabel("Compuesto")
    plt.ylabel(_etiqueta(metrica))
    plt.title(f"{_etiqueta(metrica)} por compuesto de neumático")
    plt.tight_layout()
    path = os.path.join(output_dir, f"boxplot_{metrica.lower()}_por_compuesto.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_poincare_ejemplo(speed_5hz, output_dir, lap_id, sd1, sd2):
    x1, x2 = speed_5hz[:-1], speed_5hz[1:]
    centro = np.mean(speed_5hz)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x1, x2, s=14, alpha=0.5, color="steelblue")
    ax.plot(
        [speed_5hz.min(), speed_5hz.max()], [speed_5hz.min(), speed_5hz.max()],
        linestyle="--", color="gray", label="Identidad (x$_i$=x$_{i+1}$)",
    )
    elipse = plt.matplotlib.patches.Ellipse(
        (centro, centro), width=2 * sd2, height=2 * sd1, angle=45,
        edgecolor="firebrick", facecolor="none", linewidth=2, label=f"Elipse (SD1={sd1:.1f}, SD2={sd2:.1f})",
    )
    ax.add_patch(elipse)
    ax.set_xlabel("Velocidad en t (km/h, a 5Hz)")
    ax.set_ylabel("Velocidad en t+1 (km/h, a 5Hz)")
    ax.set_title(f"Diagrama de Poincaré - vuelta {lap_id}")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    path = os.path.join(output_dir, f"poincare_vuelta_{lap_id}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_perfil_velocidad_extremos(df_telemetria, df_metricas, metrica, output_dir):
    """Compara el perfil Speed-Distance de la vuelta con mayor y menor valor de `metrica`."""
    datos = df_metricas.dropna(subset=[metrica])
    if len(datos) < 2:
        return
    lap_max = datos.loc[datos[metrica].idxmax(), "LapID"]
    lap_min = datos.loc[datos[metrica].idxmin(), "LapID"]

    plt.figure(figsize=(12, 6))
    for lap_id, estilo in [(lap_max, "Mayor"), (lap_min, "Menor")]:
        vuelta = df_telemetria[df_telemetria["LapID"] == lap_id].sort_values("Distance")
        valor = datos.loc[datos["LapID"] == lap_id, metrica].iloc[0]
        plt.plot(vuelta["Distance"], vuelta["Speed"], label=f"{estilo} {_etiqueta(metrica)} (vuelta {lap_id}, {valor:.2f})")
    plt.xlabel("Distancia (m)")
    plt.ylabel("Velocidad (km/h)")
    plt.title(f"Perfil de velocidad: vueltas extremas en {_etiqueta(metrica)}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"perfil_extremos_{metrica.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")
