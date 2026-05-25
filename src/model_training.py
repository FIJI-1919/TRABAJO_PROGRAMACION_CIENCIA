"""
model_training.py
=================
Definición y entrenamiento de modelos supervisados (clasificación y regresión)
usando pipelines de Scikit-learn.

Asignatura : Programación para la Ciencia de Datos (SCY1101)
Evaluación  : Evaluación Parcial N°2
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ──────────────────────────────────────────────
# Modelos de clasificación
# ──────────────────────────────────────────────

def get_classification_models() -> dict:
    """
    Retorna un diccionario con los pipelines base de clasificación.

    Los modelos incluidos son:
      - LogisticRegression  : modelo lineal interpretable.
      - DecisionTreeClassifier : árbol de decisión, captura no linealidades.
      - SVM                 : máquina de soporte vectorial con kernel RBF.

    Nota: El preprocesamiento (escalado, imputación) ya fue aplicado antes
    de llamar a estos pipelines (el dataset limpio se recibe directo).
    Los pipelines solo encapsulan el clasificador para mantener la interfaz
    estándar de Scikit-learn y facilitar GridSearchCV/RandomizedSearchCV.

    Retorna
    -------
    dict
        Claves: nombre del modelo (str).
        Valores: Pipeline de Scikit-learn con el clasificador.
    """
    modelos = {
        "LogisticRegression": Pipeline(steps=[
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",   # maneja desbalance de clases
            )),
        ]),

        "DecisionTreeClassifier": Pipeline(steps=[
            ("classifier", DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced",
            )),
        ]),

        "SVM": Pipeline(steps=[
            ("classifier", SVC(
                kernel="rbf",
                probability=True,          # necesario para predict_proba y ROC-AUC
                random_state=42,
                class_weight="balanced",
            )),
        ]),
    }
    return modelos


# ──────────────────────────────────────────────
# Modelos de regresión
# ──────────────────────────────────────────────

def get_regression_models() -> dict:
    """
    Retorna un diccionario con los pipelines base de regresión.

    Los modelos incluidos son:
      - LinearRegression      : regresión lineal clásica (baseline).
      - DecisionTreeRegressor : árbol de regresión, captura relaciones no lineales.

    Retorna
    -------
    dict
        Claves: nombre del modelo (str).
        Valores: Pipeline de Scikit-learn con el regresor.
    """
    modelos = {
        "LinearRegression": Pipeline(steps=[
            ("regressor", LinearRegression()),
        ]),

        "DecisionTreeRegressor": Pipeline(steps=[
            ("regressor", DecisionTreeRegressor(random_state=42)),
        ]),
    }
    return modelos


# ──────────────────────────────────────────────
# Función de entrenamiento
# ──────────────────────────────────────────────

def train_and_fit_model(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """
    Entrena un pipeline de Scikit-learn con los datos de entrenamiento.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline de Scikit-learn con el modelo a entrenar.
    X_train : array-like de shape (n_samples, n_features)
        Variables independientes de entrenamiento.
    y_train : array-like de shape (n_samples,)
        Variable objetivo de entrenamiento.

    Retorna
    -------
    Pipeline
        Pipeline ajustado (fitted).
    """
    pipeline.fit(X_train, y_train)
    return pipeline
