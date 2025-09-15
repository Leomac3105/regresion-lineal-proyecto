#!/usr/bin/env python3
"""
Regresión lineal simple con descenso de gradiente (versión mejorada).

En esta versión se incluyen métricas de evaluación para cuantificar el
rendimiento del modelo. Se genera un conjunto de datos sintético más grande y
se calcula el error cuadrático medio (MSE) y el coeficiente de
determinación R² tras el entrenamiento.

Uso: ejecutar el script con Python (por ejemplo, `python3
regresion_lineal_gd_mejorado.py`) y seguir las instrucciones en consola.
"""

import random
from typing import List, Tuple


def generate_data(n_samples: int = 500, noise_std: float = 1.0) -> Tuple[List[List[float]], List[float]]:
    """
    Genera un conjunto de datos sintético para regresión lineal.

    Cada muestra se compone de un valor x uniforme en [0, 10] y su objetivo
    correspondiente y = 5 + 2·x más ruido gaussiano.

    :param n_samples: número de muestras a generar
    :param noise_std: desviación estándar del ruido añadido
    :return: una tupla (X, y) con las características y los objetivos
    """
    random.seed(0)
    X: List[List[float]] = []
    y: List[float] = []
    for _ in range(n_samples):
        x_val = random.uniform(0, 10)
        ruido = random.gauss(0, noise_std)
        objetivo = 5.0 + 2.0 * x_val + ruido
        X.append([x_val])
        y.append(objetivo)
    return X, y


def gradient_descent_train(
    X: List[List[float]],
    y: List[float],
    alpha: float = 0.01,
    iterations: int = 10000
) -> List[float]:
    """
    Entrena un modelo de regresión lineal mediante descenso de gradiente.

    :param X: lista de vectores de características
    :param y: lista de valores objetivo
    :param alpha: tasa de aprendizaje
    :param iterations: número de iteraciones
    :return: lista de pesos aprendidos (el primer elemento es el sesgo)
    """
    n_samples = len(y)
    if n_samples == 0:
        raise ValueError("Los datos de entrenamiento están vacíos.")
    n_features = len(X[0])
    pesos: List[float] = [0.0] * (n_features + 1)
    for _ in range(iterations):
        grad: List[float] = [0.0] * (n_features + 1)
        for i in range(n_samples):
            pred = pesos[0]
            for j in range(n_features):
                pred += pesos[j+1] * X[i][j]
            error = pred - y[i]
            grad[0] += 2.0 * error
            for j in range(n_features):
                grad[j+1] += 2.0 * error * X[i][j]
        for j in range(n_features + 1):
            grad[j] /= n_samples
            pesos[j] -= alpha * grad[j]
    return pesos


def predict(features: List[float], pesos: List[float]) -> float:
    """
    Devuelve la predicción para un vector de características dado.

    :param features: valores de entrada (sin incluir el sesgo)
    :param pesos: pesos aprendidos (el primer elemento es el sesgo)
    :return: valor predicho
    """
    resultado = pesos[0]
    for j in range(len(features)):
        resultado += pesos[j+1] * features[j]
    return resultado


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """Calcula el error cuadrático medio (MSE)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    """Calcula el coeficiente de determinación R²."""
    n = len(y_true)
    if n == 0:
        return 0.0
    mean_y = sum(y_true) / n
    ss_total = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    if ss_total == 0:
        return 1.0
    return 1.0 - ss_res / ss_total


def run_demo():
    """
    Ejecuta un ejemplo de entrenamiento con un conjunto de datos sintético,
    muestra los pesos aprendidos y calcula MSE y R². Luego permite
    realizar predicciones interactivas.
    """
    X, y = generate_data(n_samples=500, noise_std=1.0)
    pesos = gradient_descent_train(X, y, alpha=0.01, iterations=10000)
    # Predicciones sobre el conjunto de entrenamiento para evaluar
    y_pred = [predict(x, pesos) for x in X]
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    b = pesos[0]
    ws = pesos[1:]
    print("\nEntrenamiento completado.")
    print(f"Sesgo (b): {b:.4f}")
    for idx, w in enumerate(ws, start=1):
        print(f"Peso w{idx}: {w:.4f}")
    print(f"\nError cuadrático medio (MSE): {mse:.4f}")
    print(f"Coeficiente de determinación (R²): {r2:.4f}")
    # Predicciones de ejemplo
    print("\nEjemplos de predicción:")
    for x_val in range(0, 11):
        y_est = predict([float(x_val)], pesos)
        print(f"x = {x_val:2d} -> y_pred = {y_est:.2f}")
    # Modo interactivo
    print("\nIntroduzca un valor de x para predecir (Enter para salir).")
    while True:
        user_input = input("x: ").strip()
        if user_input == "":
            break
        try:
            x_user = float(user_input)
            y_user_pred = predict([x_user], pesos)
            print(f"Predicción ≈ {y_user_pred:.2f}\n")
        except ValueError:
            print("Ingrese un número válido.\n")
    print("Fin.")


if __name__ == "__main__":
    run_demo()
