"""
model_evaluation.py
===================
Funciones de evaluación y comparación de modelos supervisados.

Asignatura : Programación para la Ciencia de Datos (SCY1101)
Evaluación  : Evaluación Parcial N°2
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

# Carpeta de salida para gráficos
PLOTS_DIR = os.path.join("..", "results", "plots")


def _ensure_plots_dir() -> str:
    """Crea la carpeta de gráficos si no existe y retorna su ruta."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return PLOTS_DIR


# ──────────────────────────────────────────────
# Evaluación de clasificación
# ──────────────────────────────────────────────

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> dict:
    """
    Calcula las métricas de evaluación para un clasificador binario.

    Parámetros
    ----------
    y_true : array-like
        Etiquetas reales.
    y_pred : array-like
        Etiquetas predichas.
    y_prob : array-like, opcional
        Probabilidades de la clase positiva (necesario para ROC-AUC).

    Retorna
    -------
    dict
        Diccionario con las métricas: Accuracy, Precision, Recall,
        F1-Score y ROC-AUC (si se entrega y_prob).
    """
    metricas = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metricas["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    return metricas


# ──────────────────────────────────────────────
# Evaluación de regresión
# ──────────────────────────────────────────────

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calcula las métricas de evaluación para un modelo de regresión.

    Parámetros
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.

    Retorna
    -------
    dict
        Diccionario con MAE, MSE, RMSE y R².
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "MSE":  mse,
        "RMSE": np.sqrt(mse),
        "R²":   r2_score(y_true, y_pred),
    }


# ──────────────────────────────────────────────
# Gráficos de clasificación
# ──────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nombre_modelo: str,
    save: bool = True,
) -> None:
    """
    Grafica y guarda la matriz de confusión de un clasificador.

    Parámetros
    ----------
    y_true : array-like
        Etiquetas reales.
    y_pred : array-like
        Etiquetas predichas.
    nombre_modelo : str
        Nombre del modelo (usado en el título y nombre del archivo).
    save : bool, default True
        Si True, guarda la imagen en PLOTS_DIR.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        cm, display_labels=["No Abandona", "Abandona"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusión — {nombre_modelo}", fontweight="bold")
    plt.tight_layout()

    if save:
        ruta = os.path.join(
            _ensure_plots_dir(),
            f"confusion_matrix_{nombre_modelo.lower().replace(' ', '_')}.png",
        )
        fig.savefig(ruta, dpi=150)
        print(f"  💾 Guardado: {ruta}")

    plt.show()
    plt.close(fig)


def plot_roc_curves(
    modelos: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save: bool = True,
) -> None:
    """
    Grafica las curvas ROC de múltiples clasificadores en un solo eje.

    Parámetros
    ----------
    modelos : dict
        Diccionario {nombre: pipeline_ajustado}.
    X_test : array-like
        Variables independientes del conjunto de prueba.
    y_test : array-like
        Variable objetivo del conjunto de prueba.
    save : bool, default True
        Si True, guarda la imagen en PLOTS_DIR.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colores = ["steelblue", "coral", "mediumseagreen"]

    for (nombre, modelo), color in zip(modelos.items(), colores):
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test)[:, 1]
        else:
            y_prob = modelo.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC = {auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=12)
    ax.set_title("Curvas ROC — Comparación de Clasificadores", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save:
        ruta = os.path.join(_ensure_plots_dir(), "comparativa_curvas_roc.png")
        fig.savefig(ruta, dpi=150)
        print(f"  💾 Guardado: {ruta}")

    plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# Gráficos de regresión
# ──────────────────────────────────────────────

def plot_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nombre_modelo: str,
    save: bool = True,
) -> None:
    """
    Grafica Valores Reales vs. Predichos para un regresor.

    Parámetros
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.
    nombre_modelo : str
        Nombre del modelo.
    save : bool, default True
        Si True, guarda la imagen en PLOTS_DIR.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, color="steelblue", s=15)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", lw=1.5, label="Predicción perfecta")
    ax.set_xlabel("Valores Reales")
    ax.set_ylabel("Valores Predichos")
    ax.set_title(f"Real vs. Predicho — {nombre_modelo}", fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save:
        ruta = os.path.join(
            _ensure_plots_dir(),
            f"real_vs_pred_{nombre_modelo.lower().replace(' ', '_')}.png",
        )
        fig.savefig(ruta, dpi=150)
        print(f"  💾 Guardado: {ruta}")

    plt.show()
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nombre_modelo: str,
    save: bool = True,
) -> None:
    """
    Grafica el análisis de residuos para verificar homocedasticidad.

    Parámetros
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.
    nombre_modelo : str
        Nombre del modelo.
    save : bool, default True
        Si True, guarda la imagen en PLOTS_DIR.
    """
    residuos = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Dispersión de residuos
    axes[0].scatter(y_pred, residuos, alpha=0.4, color="coral", s=15)
    axes[0].axhline(0, color="black", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Valores Predichos")
    axes[0].set_ylabel("Residuos")
    axes[0].set_title(f"Residuos vs. Predichos — {nombre_modelo}", fontweight="bold")

    # Distribución de residuos
    axes[1].hist(residuos, bins=40, color="mediumseagreen", edgecolor="white")
    axes[1].set_xlabel("Residuo")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title(f"Distribución de Residuos — {nombre_modelo}", fontweight="bold")

    plt.tight_layout()

    if save:
        ruta = os.path.join(
            _ensure_plots_dir(),
            f"residuos_{nombre_modelo.lower().replace(' ', '_')}.png",
        )
        fig.savefig(ruta, dpi=150)
        print(f"  💾 Guardado: {ruta}")

    plt.show()
    plt.close(fig)
