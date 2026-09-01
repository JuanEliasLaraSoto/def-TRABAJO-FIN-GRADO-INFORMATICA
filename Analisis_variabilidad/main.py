import argparse
import os

from perfil_variabilidad import (
    cargar_telemetria_vueltas,
    calcular_umbrales_fenotipo,
    calcular_metricas_por_vuelta,
    combinar_con_metadatos,
    PCT_BAJO,
    PCT_ALTO,
)
from metricas_variabilidad import submuestrear
from graficas_variabilidad import (
    plot_distribuciones_metricas,
    plot_metrica_vs_ritmo,
    plot_metrica_vs_tyrelife,
    plot_boxplot_por_compuesto,
    plot_poincare_ejemplo,
    plot_perfil_velocidad_extremos,
)

# Rutas fijas, mismo criterio que el resto de modulos del pipeline.
DATA_DIR = "../csv_data"
ARIMA_DATA_DIR = "../csv_data/arima"
FIGURES_DIR = "../figures/analisis_variabilidad"
TABLES_DIR = "../eda/analisis_variabilidad"

# Metricas destacadas para las graficas de correlacion con ritmo/neumatico
# (una por grupo: convencional, secuencialidad, DFA, Poincare).
METRICAS_CLAVE = ["CV_Speed_pct", "MAGE", "DFAint_alpha", "SD1"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Métricas de variabilidad del perfil de velocidad intra-vuelta "
                     "(adaptación de métricas de variabilidad glucémica, ver PLOS ONE 14(5):e0225817)."
    )
    parser.add_argument("--driver", type=str, default="HAM")
    parser.add_argument("--pct-bajo", type=float, default=PCT_BAJO)
    parser.add_argument("--pct-alto", type=float, default=PCT_ALTO)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("📡 Cargando telemetría interpolada de todas las vueltas...")
    df_telemetria = cargar_telemetria_vueltas(
        os.path.join(ARIMA_DATA_DIR, "telemetry_todas_vueltas_interpolada.csv"),
        os.path.join(ARIMA_DATA_DIR, "telemetry_todas_vueltas.csv"),
    )
    print(f"   {df_telemetria['LapID'].nunique()} vueltas, {len(df_telemetria)} puntos de telemetría.")

    umbral_bajo, umbral_alto = calcular_umbrales_fenotipo(df_telemetria, args.pct_bajo, args.pct_alto)
    print(f"🎯 Umbrales de fenotipo (percentiles {args.pct_bajo:.0f}/{args.pct_alto:.0f} de velocidad, sesión completa): "
          f"{umbral_bajo:.1f} km/h / {umbral_alto:.1f} km/h")

    print("🧮 Calculando métricas de variabilidad por vuelta...")
    df_metricas = calcular_metricas_por_vuelta(df_telemetria, umbral_bajo, umbral_alto)
    df_metricas = combinar_con_metadatos(df_metricas, os.path.join(DATA_DIR, "laps_clean.csv"), args.driver)

    tabla_path = os.path.join(TABLES_DIR, "metricas_variabilidad_por_vuelta.csv")
    df_metricas.to_csv(tabla_path, index=False)
    print(f"💾 {tabla_path}")
    print(df_metricas.describe().T.to_string())

    print("📊 Generando gráficas...")
    plot_distribuciones_metricas(df_metricas, FIGURES_DIR)

    print("🔗 Correlaciones con el tiempo de vuelta y la vida del neumático:")
    for metrica in METRICAS_CLAVE:
        r_ritmo = plot_metrica_vs_ritmo(df_metricas, metrica, FIGURES_DIR)
        r_tyre = plot_metrica_vs_tyrelife(df_metricas, metrica, FIGURES_DIR)
        if r_ritmo is not None:
            print(f"   {metrica}: r(LapTime_s)={r_ritmo:.3f}  r(TyreLife)={r_tyre:.3f}")
        plot_boxplot_por_compuesto(df_metricas, metrica, FIGURES_DIR)

    # Ejemplo de diagrama de Poincaré sobre la vuelta más rápida
    lap_rapida = df_metricas.loc[df_metricas["LapTime_s"].idxmin()]
    speed_lap_rapida = df_telemetria.loc[
        df_telemetria["LapID"] == lap_rapida["LapID"]
    ].sort_values("Time_s")["Speed"].to_numpy()
    plot_poincare_ejemplo(
        submuestrear(speed_lap_rapida), FIGURES_DIR,
        int(lap_rapida["LapID"]), lap_rapida["SD1"], lap_rapida["SD2"],
    )

    plot_perfil_velocidad_extremos(df_telemetria, df_metricas, "CV_Speed_pct", FIGURES_DIR)

    print("🎉 Análisis de variabilidad de velocidad finalizado con éxito.")
