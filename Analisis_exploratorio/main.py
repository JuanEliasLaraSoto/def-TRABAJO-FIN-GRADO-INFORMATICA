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
)
from heatmaps import heatmap_ritmo_piloto_vuelta, heatmap_correlacion_telemetria


DATA_DIR = "../csv_data"
FIGURES_DIR = "../figures/analisis_exploratorio"
TABLES_DIR = "../eda/analisis_exploratorio"


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    sns.set_theme(style="whitegrid")

    print("🔍 Iniciando análisis exploratorio...")

    laps, telemetry_3_mejores, telemetry_piloto, weather = cargar_datos_eda(DATA_DIR)

    # 1) Rendimiento por vuelta (laps_clean.csv)
    print("🏎️ Analizando tiempos de vuelta...")
    resumen_por_piloto(laps).to_csv(os.path.join(TABLES_DIR, "resumen_por_piloto.csv"), index=False)
    resumen_por_equipo(laps).to_csv(os.path.join(TABLES_DIR, "resumen_por_equipo.csv"), index=False)
    plot_boxplot_tiempos_por_piloto(laps, FIGURES_DIR)
    plot_boxplot_tiempos_por_equipo(laps, FIGURES_DIR)
    plot_evolucion_ritmo(laps, FIGURES_DIR)
    plot_efecto_compuesto(laps, FIGURES_DIR)
    plot_efecto_vida_neumatico(laps, FIGURES_DIR)
    plot_tiempos_sector(laps, FIGURES_DIR)

    # 2) Telemetría: velocidad, acelerador, freno, marchas, DRS
    print("📡 Analizando telemetría...")
    resumen_velocidad_por_piloto(telemetry_3_mejores).to_csv(
        os.path.join(TABLES_DIR, "resumen_velocidad_por_piloto.csv"), index=False
    )
    plot_perfil_velocidad(telemetry_3_mejores, FIGURES_DIR)
    plot_distribucion_velocidad(telemetry_3_mejores, FIGURES_DIR)
    plot_perfil_acelerador_freno(telemetry_piloto, FIGURES_DIR)
    plot_velocidad_maxima_por_piloto(telemetry_3_mejores, FIGURES_DIR)
    plot_uso_drs(telemetry_3_mejores, FIGURES_DIR)
    plot_distribucion_marchas(telemetry_3_mejores, FIGURES_DIR)

    # 3) Condiciones meteorológicas
    print("🌤️ Analizando condiciones meteorológicas...")
    resumen_meteorologico(weather).to_csv(os.path.join(TABLES_DIR, "resumen_meteorologico.csv"), index=False)
    plot_evolucion_meteorologia(weather, FIGURES_DIR)

    # 4) Relaciones piloto-equipo-tiempo (mapas de calor)
    print("🗺️ Generando mapas de calor...")
    heatmap_ritmo_piloto_vuelta(laps, FIGURES_DIR)
    heatmap_correlacion_telemetria(telemetry_3_mejores, FIGURES_DIR)

    print("🎉 Análisis exploratorio finalizado con éxito.")


if __name__ == "__main__":
    main()
