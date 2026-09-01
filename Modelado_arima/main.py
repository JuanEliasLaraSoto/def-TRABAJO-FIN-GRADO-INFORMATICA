import argparse
import os

import pandas as pd

from experimentos import (
    cargar_todas_vueltas_interpoladas,
    submuestrear_por_vuelta,
    añadir_icc,
    construir_series_experimento_seccion,
    construir_series_experimento_vueltas,
)
from modelo_arima import buscar_mejor_orden, evaluar_generalizacion, CANDIDATOS_ARIMA
from graficas_arima import (
    graficar_busqueda_hiperparametros,
    graficar_r2_hiperparametros,
    graficar_prediccion_vs_real,
    graficar_errores,
    graficar_comparacion_experimentos,
    graficar_acf_pacf,
)


# Rutas fijas: mismo criterio que el resto del pipeline (cada modo escribe
# siempre en el mismo sitio, no hay necesidad real de variarlas entre
# ejecuciones).
DATA_DIR = "../csv_data/arima"
TRAZADO_PATH = "../csv_data/curvatura_interpolada.csv"
FIGURES_DIR = "../figures/modelado_arima"
TABLES_DIR = "../eda/modelado_arima"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimentos de generalizacion ARIMA: division por trozo del circuito vs division por vueltas."
    )
    parser.add_argument("--frac-validacion", type=float, default=0.8)
    return parser.parse_args()


def ejecutar_experimento(nombre, serie_validacion, serie_test, args, prefijo, exog_validacion=None, exog_test=None):
    etiqueta_modelo = "ARIMAX" if exog_validacion is not None else "ARIMA"
    print(f"\n=== {nombre} ===")
    print(f"Validacion: {len(serie_validacion)} puntos | Test: {len(serie_test)} puntos")

    print(f"Buscando el mejor orden {etiqueta_modelo} sobre el conjunto de validacion...")
    mejor_orden, tabla_candidatos = buscar_mejor_orden(
        serie_validacion, CANDIDATOS_ARIMA, train_frac=args.frac_validacion, exog_validacion=exog_validacion
    )
    print(f"Mejor orden: {etiqueta_modelo}{mejor_orden}")
    print(tabla_candidatos.to_string(index=False))
    tabla_candidatos.to_csv(os.path.join(TABLES_DIR, f"{prefijo}_busqueda_hiperparametros.csv"), index=False)
    graficar_busqueda_hiperparametros(
        tabla_candidatos, FIGURES_DIR, f"{prefijo}_busqueda_hiperparametros.png",
        titulo=f"{nombre}: MAE/RMSE por orden {etiqueta_modelo} (validación)",
    )
    graficar_r2_hiperparametros(
        tabla_candidatos, FIGURES_DIR, f"{prefijo}_r2_hiperparametros.png",
        titulo=f"{nombre}: $R^2$ por orden {etiqueta_modelo} (validación)",
    )

    print("Evaluando generalizacion sobre el conjunto de test (nunca visto en la busqueda)...")
    resultado = evaluar_generalizacion(serie_validacion, serie_test, mejor_orden, exog_validacion, exog_test)
    print(f"Test final -> MAE={resultado['MAE']:.3f}  MSE={resultado['MSE']:.3f}  RMSE={resultado['RMSE']:.3f}  R2={resultado['R2']:.3f}")

    graficar_prediccion_vs_real(
        resultado, FIGURES_DIR, f"{prefijo}_prediccion_vs_real.png",
        titulo=f"{nombre}: predicción vs real en test ({etiqueta_modelo}{mejor_orden})",
    )
    graficar_errores(
        resultado, FIGURES_DIR, f"{prefijo}_errores.png",
        titulo=f"{nombre}: distribución del error de predicción (test)",
    )

    return mejor_orden, resultado


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Cargando telemetria interpolada de todas las vueltas...")
    df = cargar_todas_vueltas_interpoladas(
        os.path.join(DATA_DIR, "telemetry_todas_vueltas_interpolada.csv"),
        os.path.join(DATA_DIR, "telemetry_todas_vueltas.csv"),
    )
    df = submuestrear_por_vuelta(df)
    print(f"Vueltas disponibles: {df['LapID'].nunique()} | puntos tras submuestreo (5 Hz): {len(df)}")

    print("Añadiendo la variable exogena s(t) (ICC, indice de complejidad de curva)...")
    df = añadir_icc(df, TRAZADO_PATH)

    print("Generando correlogramas ACF/PACF de la vuelta mas rapida (diagnostico de eleccion de orden)...")
    raw = pd.read_csv(os.path.join(DATA_DIR, "telemetry_todas_vueltas.csv"))
    lap_id_mas_rapida = raw.loc[raw["LapTime_s"].idxmin(), "LapID"]
    serie_vuelta_rapida = df.loc[df["LapID"] == lap_id_mas_rapida, "Speed"].reset_index(drop=True)
    graficar_acf_pacf(
        serie_vuelta_rapida, FIGURES_DIR, "acf_pacf_velocidad.png",
        titulo="Correlogramas ACF/PACF de la velocidad (vuelta más rápida, 5 Hz)",
    )

    # Experimento 1: division espacial (un trozo del circuito para validar, otro distinto para test)
    serie_val_1, serie_test_1, exog_val_1, exog_test_1 = construir_series_experimento_seccion(df, args.frac_validacion)
    orden_1, resultado_1 = ejecutar_experimento(
        "Experimento 1 (trozo del circuito)", serie_val_1, serie_test_1, args, prefijo="exp1",
        exog_validacion=exog_val_1, exog_test=exog_test_1,
    )

    # Experimento 2: division por vueltas (misma seccion -vuelta completa-, vueltas distintas)
    serie_val_2, serie_test_2, exog_val_2, exog_test_2, vueltas_val, vueltas_test = construir_series_experimento_vueltas(df, args.frac_validacion)
    print(f"\nVueltas de validacion ({len(vueltas_val)}): {vueltas_val}")
    print(f"Vueltas de test ({len(vueltas_test)}): {vueltas_test}")
    orden_2, resultado_2 = ejecutar_experimento(
        "Experimento 2 (vueltas distintas)", serie_val_2, serie_test_2, args, prefijo="exp2",
        exog_validacion=exog_val_2, exog_test=exog_test_2,
    )

    df_resumen = pd.DataFrame([
        {"Experimento": "1. Trozo del circuito", "Mejor_orden": str(orden_1), "MAE": resultado_1["MAE"],
         "MSE": resultado_1["MSE"], "RMSE": resultado_1["RMSE"], "R2": resultado_1["R2"],
         "N_train": resultado_1["n_train"], "N_test": resultado_1["n_test"]},
        {"Experimento": "2. Vueltas distintas", "Mejor_orden": str(orden_2), "MAE": resultado_2["MAE"],
         "MSE": resultado_2["MSE"], "RMSE": resultado_2["RMSE"], "R2": resultado_2["R2"],
         "N_train": resultado_2["n_train"], "N_test": resultado_2["n_test"]},
    ])
    df_resumen.to_csv(os.path.join(TABLES_DIR, "resumen_experimentos.csv"), index=False)
    graficar_comparacion_experimentos(df_resumen, FIGURES_DIR, "comparacion_experimentos.png")

    print("\n=== Resumen final ===")
    print(df_resumen.to_string(index=False))
    print("Experimentos ARIMA completados.")
