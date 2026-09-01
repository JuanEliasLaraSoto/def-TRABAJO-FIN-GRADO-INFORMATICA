#modelo_arima.py
#ARIMA(X) con busqueda de hiperparametros (validacion) y evaluacion de
#generalizacion (test) mediante validacion walk-forward de un paso (Box-Jenkins).
#Admite opcionalmente la variable exogena s(t) (ICC, indice de complejidad
#de curva, ver experimentos.py::añadir_icc): pasar `exog` convierte el ARIMA
#en un ARIMAX sin cambiar de clase, ya que `statsmodels.tsa.arima.model.ARIMA`
#soporta regresores exogenos de forma nativa (esta implementada internamente
#sobre el mismo motor de espacio de estados que SARIMAX).
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

ORDEN_ARIMA = (5, 1, 0)

CANDIDATOS_ARIMA = [
    (2, 1, 0),
    (3, 1, 0),
    (5, 1, 0),
    (8, 1, 0),
    (5, 1, 1),
    (5, 0, 0),
]


def _walk_forward(train, test, order, exog_train=None, exog_test=None):
    """
    Ajusta un ARIMA(p,d,q) (o ARIMAX si se pasa `exog_train`/`exog_test`,
    la variable s(t) de dificultad del trazado) una única vez sobre `train`
    y genera predicciones walk-forward de un paso (Steps=1) sobre `test`,
    usando `extend()` en lugar de un bucle de `append()`: `extend()`
    continúa el filtro de Kalman desde el estado final de `train` y filtra
    TODO `test` en una única pasada O(n_test), produciendo en
    `fittedvalues` la predicción a un paso en cada instante (basada solo en
    datos anteriores a ese instante, incluido el valor de `exog` de ese
    mismo instante) sin reestimar los parámetros. Es matemáticamente
    equivalente a predecir y reinyectar el dato real punto a punto, pero
    evita el coste cuadrático de volver a filtrar toda la serie acumulada
    en cada `append()`.
    """
    # `extend()` exige que el indice de `test` continue el de `train`
    # (necesario cuando train/test vienen de series construidas por
    # separado, p. ej. en evaluar_generalizacion, ambas indexadas desde 0).
    train = pd.Series(train).reset_index(drop=True)
    test = pd.Series(np.asarray(test))
    test.index = range(len(train), len(train) + len(test))

    if exog_train is not None:
        exog_train = pd.Series(np.asarray(exog_train)).reset_index(drop=True)
        exog_test = pd.Series(np.asarray(exog_test))
        exog_test.index = test.index

        # El ICC (curvatura, ~1e-4) y la velocidad (~1e2 km/h) difieren en
        # varios ordenes de magnitud; sin tipificar, el optimizador MLE de
        # statsmodels se vuelve numericamente inestable (se observaron
        # errores de descomposicion LU y ajustes divergentes en algunos
        # ordenes candidatos). Se tipifica con la media/desviacion del
        # propio `exog_train` de cada split, nunca con estadisticos del
        # conjunto de test.
        media, desviacion = exog_train.mean(), exog_train.std()
        if desviacion > 0:
            exog_train = (exog_train - media) / desviacion
            exog_test = (exog_test - media) / desviacion

    resultado = ARIMA(train, exog=exog_train, order=order).fit()
    resultado_extendido = resultado.extend(test, exog=exog_test)

    predicciones = resultado_extendido.fittedvalues.to_numpy()
    reales = test.to_numpy()

    errores = predicciones - reales
    mae = float(np.mean(np.abs(errores)))
    mse = float(np.mean(errores ** 2))
    rmse = float(np.sqrt(mse))

    # Coeficiente de determinación R^2 = 1 - SS_res/SS_tot, comparando el
    # error del modelo frente al de predecir siempre la media del propio
    # conjunto de test (criterio de calidad pedido explícitamente por Ezequiel).
    ss_res = float(np.sum(errores ** 2))
    ss_tot = float(np.sum((reales - reales.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
        "n_train": len(train), "n_test": len(test),
        "predicciones": predicciones, "reales": reales,
    }


def evaluar_walk_forward(serie, order=ORDEN_ARIMA, train_frac=0.8, exog=None):
    """
    Divide `serie` cronológicamente en train_frac/1-train_frac y evalúa
    mediante validación walk-forward (ver `_walk_forward`). Si se pasa
    `exog` (la variable s(t) de dificultad, alineada punto a punto con
    `serie`), se divide con el mismo corte y se usa como regresor exógeno.
    """
    n = len(serie)
    n_train = int(n * train_frac)
    if n_train < 20 or n - n_train < 5:
        raise ValueError("Serie demasiado corta para el split train/test solicitado.")

    train, test = serie.iloc[:n_train], serie.iloc[n_train:]
    if exog is not None:
        exog_train, exog_test = exog.iloc[:n_train], exog.iloc[n_train:]
    else:
        exog_train = exog_test = None
    return _walk_forward(train, test, order, exog_train, exog_test)


def buscar_mejor_orden(serie_validacion, candidatos=CANDIDATOS_ARIMA, train_frac=0.8, exog_validacion=None):
    """
    Búsqueda de hiperparámetros: evalúa cada orden ARIMA candidato mediante
    walk-forward *dentro* de `serie_validacion` (nunca toca el conjunto de
    test), y selecciona el de menor RMSE. `exog_validacion`, si se pasa, es
    la variable s(t) de dificultad usada como regresor exógeno (ARIMAX).

    Devuelve (mejor_orden, DataFrame con las métricas de cada candidato).
    """
    filas = []
    for order in candidatos:
        try:
            metricas = evaluar_walk_forward(serie_validacion, order=order, train_frac=train_frac, exog=exog_validacion)
            filas.append({
                "Orden": str(order), "MAE": metricas["MAE"], "MSE": metricas["MSE"],
                "RMSE": metricas["RMSE"], "R2": metricas["R2"],
            })
        except Exception as e:
            print(f"   Orden {order} descartado: {e}")

    df_resultados = pd.DataFrame(filas).sort_values("RMSE").reset_index(drop=True)
    mejor_orden = candidatos[[str(o) for o in candidatos].index(df_resultados.iloc[0]["Orden"])]
    return mejor_orden, df_resultados


def evaluar_generalizacion(serie_train, serie_test, order, exog_train=None, exog_test=None):
    """
    Ajusta el ARIMA(order) elegido sobre TODA `serie_train` (p. ej. el
    conjunto de validación completo) y evalúa su capacidad de generalización
    mediante walk-forward sobre `serie_test`, una porción de datos distinta
    (otro tramo del circuito, u otras vueltas) nunca vista durante la
    selección de hiperparámetros. `exog_train`/`exog_test`, si se pasan, son
    la variable s(t) de dificultad usada como regresor exógeno (ARIMAX).
    """
    return _walk_forward(serie_train, serie_test, order, exog_train, exog_test)
