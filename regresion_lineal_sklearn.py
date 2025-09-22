"""
Implementación de regresión lineal utilizando un framework de aprendizaje automático.

Este script utiliza la biblioteca ``scikit‑learn`` para entrenar un modelo de
regresión lineal sobre un conjunto de datos sintético. Además de entrenar el
modelo, calcula el error cuadrático medio (MSE) y el coeficiente de
determinación (R²) en conjuntos de entrenamiento y prueba. Al final se
proporciona un modo interactivo para hacer predicciones con nuevos valores
de la variable independiente.


"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class RegressionResults:
    """Estructura para almacenar los resultados de la regresión."""
    coeficiente: float
    intercepto: float
    mse_entrenamiento: float
    mse_prueba: float
    r2_entrenamiento: float
    r2_prueba: float


def generar_datos(n: int = 200, ruido_std: float = 5.0, semilla: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Genera un conjunto de datos sintético para regresión lineal.

    Genera ``n`` muestras de datos ``x`` uniformemente distribuidos entre 0 y 10,
    y valores de ``y`` que siguen la relación lineal ``y = 3 + 4x + ruido``.

    Args:
        n: número de muestras a generar.
        ruido_std: desviación estándar del ruido gaussiano.
        semilla: semilla para reproducibilidad.

    Returns:
        Una tupla ``(X, y)`` donde ``X`` es un arreglo bidimensional de forma
        (n, 1) y ``y`` un arreglo unidimensional de forma (n,).
    """
    rng = np.random.default_rng(semilla)
    x = rng.uniform(0, 10, n)
    ruido = rng.normal(0, ruido_std, n)
    y = 3 + 4 * x + ruido
    # scikit‑learn espera X en forma 2D, incluso con una sola característica
    return x.reshape(-1, 1), y


def entrenar_regresion_lineal(
    X: np.ndarray,
    y: np.ndarray,
    proporción_prueba: float = 0.2,
    semilla: int = 0,
) -> Tuple[LinearRegression, RegressionResults, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Entrena un modelo de regresión lineal usando scikit‑learn y evalúa su desempeño.

    Se divide el conjunto de datos en entrenamiento y prueba, se entrena un
    ``LinearRegression`` sobre el conjunto de entrenamiento y se calculan
    métricas de error y R² en ambos conjuntos.

    Args:
        X: matriz de características de entrada de forma (n_samples, n_features).
        y: vector de valores objetivo de forma (n_samples,).
        proporción_prueba: proporción de datos reservada para la prueba.
        semilla: semilla para la división aleatoria.

    Returns:
        Una tupla con el modelo entrenado, un ``RegressionResults`` con las
        métricas, y las divisiones ``X_train``, ``X_test``, ``y_train``, ``y_test``.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=proporción_prueba, random_state=semilla
    )
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # Cálculo de métricas
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    resultados = RegressionResults(
        coeficiente=float(modelo.coef_[0]),
        intercepto=float(modelo.intercept_),
        mse_entrenamiento=mse_train,
        mse_prueba=mse_test,
        r2_entrenamiento=r2_train,
        r2_prueba=r2_test,
    )

    return modelo, resultados, X_train, X_test, y_train, y_test


def modo_interactivo(modelo: LinearRegression) -> None:
    """Permite introducir valores de x para obtener predicciones con el modelo entrenado.

    El usuario puede escribir valores separados por comas o espacios. Se finaliza
    introduciendo una línea vacía. Para entornos en los que no se dispone de
    entrada estándar (por ejemplo, al ejecutar en notebooks sin consola), este
    modo se puede omitir.
    """
    print("\nModo interactivo: introduce valores de x (separados por comas o espacios) o presiona Enter para salir.")
    while True:
        try:
            linea = input("x = ").strip()
        except EOFError:
            # En entornos sin entrada estándar se ignora el modo interactivo
            print("\nEntrada no disponible. Saliendo del modo interactivo.")
            break
        if linea == "":
            print("Saliendo del modo interactivo.")
            break
        # Extraer números de la línea
        try:
            valores = [float(v) for v in linea.replace(",", " ").split()]
            if not valores:
                raise ValueError
        except ValueError:
            print("Entrada no válida. Introduce números separados por espacios o comas.")
            continue
        datos = np.array(valores).reshape(-1, 1)
        predicciones = modelo.predict(datos)
        for x_val, y_pred in zip(valores, predicciones):
            print(f"x = {x_val:.3f} -> y_predicho = {y_pred:.3f}")


def main(argv: list[str] | None = None) -> int:
    """Función principal del script.

    Genera datos sintéticos, entrena el modelo y muestra métricas. Si se
    proporciona el argumento ``--no-interactivo`` se omite el modo
    interactivo.
    """
    if argv is None:
        argv = sys.argv[1:]
    modo_interactivo_habilitado = True
    if "--no-interactivo" in argv:
        modo_interactivo_habilitado = False

    # Generar datos sintéticos
    X, y = generar_datos(n=300, ruido_std=3.0, semilla=0)

    # Entrenar y evaluar el modelo
    modelo, resultados, X_train, X_test, y_train, y_test = entrenar_regresion_lineal(X, y)

    # Mostrar resultados
    print("\n=== Resultados de la regresión lineal con scikit‑learn ===")
    print(f"Coeficiente (pendiente): {resultados.coeficiente:.3f}")
    print(f"Intercepto: {resultados.intercepto:.3f}")
    print(f"MSE entrenamiento: {resultados.mse_entrenamiento:.3f}")
    print(f"MSE prueba: {resultados.mse_prueba:.3f}")
    print(f"R² entrenamiento: {resultados.r2_entrenamiento:.3f}")
    print(f"R² prueba: {resultados.r2_prueba:.3f}\n")

    # Modo interactivo
    if modo_interactivo_habilitado:
        modo_interactivo(modelo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
