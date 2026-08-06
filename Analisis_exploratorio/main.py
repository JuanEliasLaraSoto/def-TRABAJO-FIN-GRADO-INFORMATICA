import argparse
import os

import seaborn as sns

from cargar_datos import cargar_datos_eda
from estadisticas_vueltas import (
    resumen_por_piloto,
    resumen_por_equipo,
    plot_boxplot_tiempos_por_piloto,
    plot_boxplot_tiempos_por_equipo,
    plot_evolucion_ritmo,
    plot_efecto_compuesto,
    plot_efecto_vida_neumatico,
    plot_tiempos_sector,
)
from estadisticas_telemetria import (
    resumen_velocidad_por_piloto,
    plot_perfil_velocidad,
    plot_distribucion_velocidad,
    plot_perfil_acelerador_freno,
    plot_velocidad_maxima_por_piloto,
    plot_uso_drs,
    plot_distribucion_marchas,
)
from estadisticas_meteorologicas import (
    resumen_meteorologico,
    plot_evolucion_meteorologia,
    correlacion_meteorologia_ritmo,
)
from heatmaps import heatmap_ritmo_piloto_vuelta, heatmap_correlacion_telemetria


def parse_args():
    parser = argparse.ArgumentParser(
        description="Análisis exploratorio de los datos de F1 extraídos y limpiados."
    )
    parser.add_argument("--data-dir", type=str, default="../csv_data", help="Directorio con los CSV de entrada")
    parser.add_argument("--figures-dir", type=str, default="../figures", help="Directorio de salida para las figuras")
    parser.add_argument("--tables-dir", type=str, default="../csv_data/eda", help="Directorio de salida para las tablas resumen")
    return parser.parse_args()


args = parse_args()
os.makedirs(args.figures_dir, exist_ok=True)
os.makedirs(args.tables_dir, exist_ok=True)

sns.set_theme(style="whitegrid")

print("🔍 Iniciando análisis exploratorio...")

laps, telemetry_3_mejores, telemetry_piloto, weather = cargar_datos_eda(args.data_dir)

# 1) Rendimiento por vuelta (laps_clean.csv)
print("🏎️ Analizando tiempos de vuelta...")
resumen_por_piloto(laps).to_csv(os.path.join(args.tables_dir, "resumen_por_piloto.csv"), index=False)
resumen_por_equipo(laps).to_csv(os.path.join(args.tables_dir, "resumen_por_equipo.csv"), index=False)
plot_boxplot_tiempos_por_piloto(laps, args.figures_dir)
plot_boxplot_tiempos_por_equipo(laps, args.figures_dir)
plot_evolucion_ritmo(laps, args.figures_dir)
plot_efecto_compuesto(laps, args.figures_dir)
plot_efecto_vida_neumatico(laps, args.figures_dir)
plot_tiempos_sector(laps, args.figures_dir)

# 2) Telemetría: velocidad, acelerador, freno, marchas, DRS
print("📡 Analizando telemetría...")
resumen_velocidad_por_piloto(telemetry_3_mejores).to_csv(
    os.path.join(args.tables_dir, "resumen_velocidad_por_piloto.csv"), index=False
)
plot_perfil_velocidad(telemetry_3_mejores, args.figures_dir)
plot_distribucion_velocidad(telemetry_3_mejores, args.figures_dir)
plot_perfil_acelerador_freno(telemetry_piloto, args.figures_dir)
plot_velocidad_maxima_por_piloto(telemetry_3_mejores, args.figures_dir)
plot_uso_drs(telemetry_3_mejores, args.figures_dir)
plot_distribucion_marchas(telemetry_3_mejores, args.figures_dir)

# 3) Condiciones meteorológicas
print("🌤️ Analizando condiciones meteorológicas...")
resumen_meteorologico(weather).to_csv(os.path.join(args.tables_dir, "resumen_meteorologico.csv"), index=False)
plot_evolucion_meteorologia(weather, args.figures_dir)
correlacion = correlacion_meteorologia_ritmo(laps, weather, args.figures_dir)
if correlacion is not None:
    print(f"🌡️ Correlación aproximada temperatura de pista / tiempo de vuelta: {correlacion:.3f}")

# 4) Relaciones piloto-equipo-tiempo (mapas de calor)
print("🗺️ Generando mapas de calor...")
heatmap_ritmo_piloto_vuelta(laps, args.figures_dir)
heatmap_correlacion_telemetria(telemetry_3_mejores, args.figures_dir)

print("🎉 Análisis exploratorio finalizado con éxito.")
