#graficas_arima.py
#graficas de apoyo para los experimentos ARIMA: busqueda de hiperparametros,
#prediccion vs real, y comparacion final entre experimentos
import os

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def graficar_busqueda_hiperparametros(tabla_candidatos, output_dir, filename, titulo):
    """Barras de MAE/RMSE por cada orden ARIMA candidato evaluado en validación."""
    tabla = tabla_candidatos.sort_values("RMSE")
    x = np.arange(len(tabla))

    fig, ax = plt.subplots(figsize=(9, 5))
    ancho = 0.35
    ax.bar(x - ancho / 2, tabla["MAE"], width=ancho, label="MAE", color="#1f77b4")
    ax.bar(x + ancho / 2, tabla["RMSE"], width=ancho, label="RMSE", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(tabla["Orden"], rotation=30, ha="right")
    ax.set_ylabel("Error (km/h)")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def graficar_r2_hiperparametros(tabla_candidatos, output_dir, filename, titulo):
    """Barras del coeficiente de determinación R^2 por cada orden ARIMA candidato."""
    tabla = tabla_candidatos.sort_values("RMSE")
    x = np.arange(len(tabla))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, tabla["R2"], color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tabla["Orden"], rotation=30, ha="right")
    ax.set_ylabel("$R^2$")
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def graficar_prediccion_vs_real(resultado, output_dir, filename, titulo, n_puntos=300):
    """Serie temporal de predicción walk-forward vs valor real en el conjunto de test (primeros n_puntos, para legibilidad)."""
    reales = resultado["reales"][:n_puntos]
    predicciones = resultado["predicciones"][:n_puntos]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(reales, label="Velocidad real", color="black", linewidth=1.2)
    ax.plot(predicciones, label="Predicción ARIMA (1 paso)", color="#d62728", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Índice temporal en el conjunto de test")
    ax.set_ylabel("Velocidad (km/h)")
    ax.set_title(f"{titulo}\n(primeros {min(n_puntos, len(resultado['reales']))} de {len(resultado['reales'])} puntos de test)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def graficar_errores(resultado, output_dir, filename, titulo):
    """Histograma del error de predicción (predicción - real) en el conjunto de test."""
    errores = resultado["predicciones"] - resultado["reales"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errores, bins=40, color="#9467bd", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Error de predicción (km/h)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def graficar_acf_pacf(serie, output_dir, filename, titulo, lags=40):
    """
    Correlogramas de autocorrelación (ACF) y autocorrelación parcial (PACF)
    de una serie de velocidad a 5 Hz, para ilustrar por qué se descartó la
    identificación visual clásica de (p, d, q): a diferencia de los
    ejemplos de manual, no se observa un corte abrupto y limpio en ninguno
    de los dos correlogramas.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(serie, lags=lags, ax=ax1, color="#1f77b4")
    ax1.set_title("ACF")
    ax1.set_xlabel("Rezago (muestras a 5 Hz)")
    plot_pacf(serie, lags=lags, ax=ax2, method="ywm", color="#d62728")
    ax2.set_title("PACF")
    ax2.set_xlabel("Rezago (muestras a 5 Hz)")
    fig.suptitle(titulo)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")


def graficar_comparacion_experimentos(df_resumen, output_dir, filename):
    """Barras comparando MAE, RMSE y R^2 finales (en test) entre los dos experimentos."""
    x = np.arange(len(df_resumen))
    ancho = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x - ancho / 2, df_resumen["MAE"], width=ancho, label="MAE", color="#1f77b4")
    ax1.bar(x + ancho / 2, df_resumen["RMSE"], width=ancho, label="RMSE", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_resumen["Experimento"])
    ax1.set_ylabel("Error en test (km/h)")
    ax1.set_title("MAE / RMSE en test")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, df_resumen["R2"], color="#2ca02c")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_resumen["Experimento"])
    ax2.set_ylabel("$R^2$")
    ax2.set_title("Coeficiente de determinación en test")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Comparación final entre experimentos de generalización")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")
