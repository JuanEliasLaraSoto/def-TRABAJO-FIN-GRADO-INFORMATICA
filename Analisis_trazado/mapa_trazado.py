#mapa_trazado.py
#dibuja el trazado coloreado por dificultad geometrica, y las curvas numeradas
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

COLOR_DIFICULTAD = {"Baja": "#2ca02c", "Media": "#ff9f1c", "Alta": "#d62728"}


def plot_mapa_curvatura(spline_track, output_dir, radio_clip=600.0):
    """
    Dibuja el trazado completo coloreado de forma continua según la
    dificultad geométrica (basada en el radio de curvatura local, acotado a
    radio_clip para dar contraste visual entre curvas cerradas y rectas).
    """
    x = spline_track["X"].values
    y = spline_track["Y"].values
    radio = np.clip(spline_track["Radius_m"].values, 0, radio_clip)
    # Dificultad normalizada: 0 = recta (radio >= radio_clip), 1 = curva muy cerrada
    dificultad = 1.0 - (radio / radio_clip)

    puntos = np.array([x, y]).T.reshape(-1, 1, 2)
    segmentos = np.concatenate([puntos[:-1], puntos[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(12, 9))
    lc = LineCollection(segmentos, cmap="RdYlGn_r", linewidths=6)
    lc.set_array(dificultad[:-1])
    ax.add_collection(lc)
    ax.set_xlim(x.min() - 100, x.max() + 100)
    ax.set_ylim(y.min() - 100, y.max() + 100)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Mapa de dificultad del trazado (radio de curvatura local)")
    cbar = fig.colorbar(lc, ax=ax, shrink=0.7)
    cbar.set_label("Dificultad (0 = recta, 1 = curva cerrada)")
    plt.tight_layout()
    path = os.path.join(output_dir, "mapa_trazado_dificultad.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"🗺️ {path}")


def plot_mapa_curvas_numeradas(spline_track, curvas, output_dir, nombres=None):
    """
    Dibuja el trazado y superpone marcadores numerados en el ápex de cada
    curva detectada, coloreados según su categoría de dificultad. Si se
    proporciona `nombres` (dict Numero -> nombre real de la curva), añade
    una etiqueta con línea guía por cada nombre distinto, desplazada hacia
    fuera del trazado para evitar solapar los marcadores más próximos entre
    sí (p. ej. las dos curvas de una misma chicane).
    """
    x = spline_track["X"].values
    y = spline_track["Y"].values
    nombres = nombres or {}

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.plot(x, y, color="black", linewidth=2, zorder=1)
    ax.set_xlim(x.min() - 350, x.max() + 350)
    ax.set_ylim(y.min() - 350, y.max() + 500)

    x_centro, y_centro = x.mean(), y.mean()
    nombres_colocados = set()

    for curva in curvas:
        color = COLOR_DIFICULTAD.get(curva["Dificultad"], "gray")
        ax.scatter(
            curva["X_apex"], curva["Y_apex"], s=220, color=color,
            edgecolors="black", zorder=3,
        )
        ax.annotate(
            str(curva["Numero"]), (curva["X_apex"], curva["Y_apex"]),
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white", zorder=4,
        )

        nombre = nombres.get(curva["Numero"])
        if nombre and nombre not in nombres_colocados:
            nombres_colocados.add(nombre)
            dx, dy = curva["X_apex"] - x_centro, curva["Y_apex"] - y_centro
            norma = np.hypot(dx, dy) or 1.0
            desplazamiento = 320.0
            xt = curva["X_apex"] + dx / norma * desplazamiento
            yt = curva["Y_apex"] + dy / norma * desplazamiento
            ax.annotate(
                nombre,
                (curva["X_apex"], curva["Y_apex"]),
                xytext=(xt, yt),
                ha="center", va="center", fontsize=10,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
                zorder=2,
            )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=12, label=d)
        for d, c in COLOR_DIFICULTAD.items()
    ]
    ax.legend(handles=handles, title="Dificultad", loc="lower right")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Curvas identificadas y su dificultad")
    plt.tight_layout()
    path = os.path.join(output_dir, "mapa_trazado_curvas_numeradas.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"🗺️ {path}")
