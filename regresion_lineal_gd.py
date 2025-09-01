#!/usr/bin/env python3
"""
Regresión lineal simple con descenso de gradiente.

Este script genera un conjunto de datos sintético y ajusta un modelo de
regresión lineal usando descenso de gradiente sin recurrir a bibliotecas de
aprendizaje automático. Al ejecutarse, muestra los parámetros aprendidos y
permite realizar predicciones desde la consola.
"""

import random
from typing import List, Tuple

def generate_data(n_samples: int = 100, noise_std: float = 1.0) -> Tuple[List[List[float]], List[float]]:
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
    """Devuelve la predicción para un vector de características dado."""
    resultado = pesos[0]
    for j in range(len(features)):
        resultado += pesos[j+1] * features[j]
    return resultado

def run_demo():
    """Ejecuta un ejemplo de entrenamiento y permite hacer predicciones."""
    X, y = generate_data(n_samples=100, noise_std=1.0)
    pesos = gradient_descent_train(X, y, alpha=0.01, iterations=10000)
    b = pesos[0]
    ws = pesos[1:]
    print("\nEntrenamiento completado.")
    print(f"Sesgo (b): {b:.4f}")
    for idx, w in enumerate(ws, start=1):
        print(f"Peso w{idx}: {w:.4f}")
    print("\nEjemplos de predicción:")
    for x_val in range(0, 11):
        y_pred = predict([float(x_val)], pesos)
        print(f"x = {x_val:2d} -> y_pred = {y_pred:.2f}")
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
