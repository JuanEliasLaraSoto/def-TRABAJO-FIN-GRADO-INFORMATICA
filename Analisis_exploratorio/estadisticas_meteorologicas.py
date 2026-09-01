#estadisticas_meteorologicas.py
#estadisticas y graficas de condiciones meteorologicas
import os

import matplotlib.pyplot as plt
import pandas as pd


def _time_a_segundos(df, col="Time"):
    df = df.copy()
    df[col + "_s"] = pd.to_timedelta(df[col]).dt.total_seconds()
    return df


def resumen_meteorologico(weather):
    cols = [c for c in ["AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Pressure"] if c in weather.columns]
    return weather[cols].describe().transpose().reset_index().rename(columns={"index": "Variable"})


def plot_evolucion_meteorologia(weather, output_dir):
    weather = _time_a_segundos(weather)
    variables = [c for c in ["AirTemp", "TrackTemp", "Humidity", "WindSpeed"] if c in weather.columns]
    if not variables:
        return
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 3 * len(variables)), sharex=True)
    if len(variables) == 1:
        axes = [axes]
    for ax, var in zip(axes, variables):
        ax.plot(weather["Time_s"], weather[var])
        ax.set_ylabel(var)
    axes[-1].set_xlabel("Tiempo de sesión (s)")
    fig.suptitle("Evolución de las condiciones meteorológicas durante la sesión")
    plt.tight_layout()
    path = os.path.join(output_dir, "evolucion_meteorologia.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")
