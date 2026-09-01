#mapa_trazado.py
#dibuja el trazado coloreado por dificultad geometrica, y las curvas numeradas
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

COLOR_DIFICULTAD = {"Baja": "#2ca02c", "Media": "#ff9f1c", "Alta": "#d62728"}


def _paleta_dificultad():
    """
    Paleta secuencial violeta-morado (familia distinta del naranja/rojo ya
    usado en otras figuras, y sin amarillo, que se confunde con el fondo
    blanco) y recortada en su extremo más claro: incluso el valor de
    dificultad mínima (recta) queda en un lila claro pero visible, no un
    color casi blanco.
    """
    base = plt.get_cmap("RdPu")
    return LinearSegmentedColormap.from_list("dificultad_trazado", base(np.linspace(0.25, 1.0, 256)))


def plot_relacion_radio_velocidad(resumen, output_dir, r=None):
    """
    Dispersión del radio mínimo de curvatura (dificultad geométrica) frente a
    la velocidad mínima real de paso por curva (telemetría del piloto),
    coloreada por categoría de dificultad, con el coeficiente de correlación
    de Pearson entre ambas variables.

    Si no se proporciona `r`, se calcula aquí mismo (Pearson) sobre las
    curvas con datos válidos.
    """
    datos = resumen.dropna(subset=["Radio_min_m", "Velocidad_min_kmh"])
    if datos.empty:
        print("⚠️ Sin datos suficientes para el gráfico radio vs velocidad.")
        return

    if r is None and len(datos) >= 2:
        r = datos["Radio_min_m"].corr(datos["Velocidad_min_kmh"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for dificultad, color in COLOR_DIFICULTAD.items():
        subset = datos[datos["Dificultad"] == dificultad]
        ax.scatter(
            subset["Radio_min_m"], subset["Velocidad_min_kmh"],
            s=110, color=color, edgecolors="black", label=dificultad, zorder=3,
        )

    for _, fila in datos.iterrows():
        ax.annotate(
            str(int(fila["Numero"])), (fila["Radio_min_m"], fila["Velocidad_min_kmh"]),
            textcoords="offset points", xytext=(6, 6), fontsize=9,
        )

    if r is not None:
        ax.set_title(f"Radio de curvatura vs velocidad mínima real por curva (r = {r:.2f})")
    else:
        ax.set_title("Radio de curvatura vs velocidad mínima real por curva")

    ax.set_xlabel("Radio mínimo de curvatura (m)")
    ax.set_ylabel("Velocidad mínima real (km/h)")
    ax.grid(alpha=0.3)
    ax.legend(title="Dificultad")
    plt.tight_layout()
    path = os.path.join(output_dir, "relacion_radio_velocidad.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def plot_mapa_curvatura(spline_track, output_dir, radio_clip=600.0, filename="mapa_trazado_dificultad.png", titulo=None):
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
    # Paleta secuencial (no divergente): recta = color suave, curva muy
    # cerrada = color muy oscuro. La primera versión (RdYlGn_r) apenas se
    # apreciaba porque el rojo y el verde tienen luminosidad similar; la
    # segunda (YlOrRd) se resolvió, pero su extremo amarillo pálido se
    # confundía con el fondo blanco de la figura.
    lc = LineCollection(segmentos, cmap=_paleta_dificultad(), linewidths=6)
    lc.set_array(dificultad[:-1])
    ax.add_collection(lc)
    ax.set_xlim(x.min() - 100, x.max() + 100)
    ax.set_ylim(y.min() - 100, y.max() + 100)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titulo or "Mapa de dificultad del trazado (radio de curvatura local)")
    cbar = fig.colorbar(lc, ax=ax, shrink=0.7)
    cbar.set_label("Dificultad (0 = recta, 1 = curva cerrada)")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


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
                ha="center", va="center", fontsize=15, fontweight="bold",
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
    print(f"📊 {path}")
