# Proyecto de regresión lineal con descenso de gradiente

Este repositorio contiene implementaciones manuales de la **regresión lineal simple** utilizando el algoritmo de **descenso de gradiente**. El objetivo es comprender y programar paso a paso el algoritmo sin emplear bibliotecas de aprendizaje automático como scikit‑learn.

## Archivos

- **`regresion_lineal_gd.py`**: versión inicial del algoritmo con un conjunto de datos sintético y modo interactivo de predicción.
- **`regresion_lineal_gd_mejorado.py`**: versión mejorada que incluye métricas de evaluación (MSE y R²) y utiliza un conjunto de datos sintético más grande.

## Cómo ejecutar

1. Instala Python 3 (por ejemplo, Python 3.8 o superior).
2. Clona este repositorio o descarga los archivos:
   ```bash
   git clone https://github.com/Leomac3105/regresion-lineal-proyecto.git
   cd regresion-lineal-proyecto
   ```
3. Ejecuta el script deseado. Por ejemplo:
   ```bash
   python3 regresion_lineal_gd_mejorado.py
   ```
   El programa entrenará el modelo, mostrará los parámetros aprendidos y las métricas de evaluación, y luego te permitirá introducir valores de `x` para obtener predicciones.

## Resultados esperados

Al correr `regresion_lineal_gd_mejorado.py` se imprimen en consola:

- Los parámetros aprendidos (`b` y `w1`).
- El **error cuadrático medio (MSE)**.
- El **coeficiente de determinación (R²)**.
- Ejemplos de predicciones para valores de `x` entre 0 y 10.
- Un modo interactivo para predecir `y` a partir de un valor de `x`.

Estas métricas te permitirán cuantificar qué tan bien está aprendiendo el modelo sobre el conjunto de datos utilizado.

## Próximos pasos

Para la siguiente entrega se recomienda experimentar con datasets reales (por ejemplo, conjuntos de datos disponibles en el [UCI Machine Learning Repository](https://archive.ics.uci.edu)) y comparar los resultados con implementaciones de frameworks como scikit‑learn.
