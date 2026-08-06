#estadisticas_meteorologicas.py
#estadisticas y graficas de condiciones meteorologicas, y su relacion con el ritmo de carrera
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def correlacion_meteorologia_ritmo(laps, weather, output_dir):
    """
    Aproxima el instante de cada vuelta como el tiempo de sesión acumulado
    (suma de LapTime_s de las vueltas previas del mismo piloto) y cruza esa
    marca temporal con la lectura meteorológica más cercana en el tiempo,
    para estudiar si la temperatura de pista se relaciona con el ritmo de
    carrera. Es una aproximación: no se dispone del instante de inicio real
    de cada vuelta en laps_clean.csv.
    """
    if "TrackTemp" not in weather.columns:
        return None

    laps = laps.sort_values(["Driver", "LapNumber"]).copy()
    laps["Time_acumulado_s"] = laps.groupby("Driver")["LapTime_s"].cumsum()

    weather = _time_a_segundos(weather).sort_values("Time_s")
    laps_ordenado = laps.sort_values("Time_acumulado_s")

    combinado = pd.merge_asof(
        laps_ordenado,
        weather[["Time_s", "TrackTemp"]],
        left_on="Time_acumulado_s",
        right_on="Time_s",
        direction="nearest",
    )

    plt.figure(figsize=(8, 6))
    sns.regplot(data=combinado, x="TrackTemp", y="LapTime_s", scatter_kws={"alpha": 0.4})
    plt.xlabel("Temperatura de pista (°C, aprox. en el instante de la vuelta)")
    plt.ylabel("Tiempo de vuelta (s)")
    plt.title("Relación aproximada entre temperatura de pista y ritmo de vuelta")
    plt.tight_layout()
    path = os.path.join(output_dir, "scatter_temperatura_vs_ritmo.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 {path}")

    return combinado[["TrackTemp", "LapTime_s"]].corr().iloc[0, 1]
