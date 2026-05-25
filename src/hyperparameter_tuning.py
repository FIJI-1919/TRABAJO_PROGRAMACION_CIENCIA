"""
hyperparameter_tuning.py
========================
Funciones para la optimización de hiperparámetros usando
GridSearchCV y RandomizedSearchCV, con serialización de modelos.

Asignatura : Programación para la Ciencia de Datos (SCY1101)
Evaluación  : Evaluación Parcial N°2
"""

import os

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

# Carpeta por defecto para guardar modelos
DEFAULT_MODELS_DIR = os.path.join("..", "models", "trained_models")


# ──────────────────────────────────────────────
# Optimización de clasificadores
# ──────────────────────────────────────────────

def tune_classification_model(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "grid",
    n_iter: int = 10,
    cv: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
) -> Pipeline:
    """
    Optimiza hiperparámetros de un pipeline de clasificación.

    Usa GridSearchCV (búsqueda exhaustiva) o RandomizedSearchCV
    (búsqueda aleatoria eficiente) con validación cruzada estratificada
    (StratifiedKFold) para mantener la proporción de clases en cada fold.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline base de Scikit-learn con el clasificador.
    param_grid : dict
        Grilla o distribución de hiperparámetros a explorar.
        Las claves deben seguir la convención 'paso__parametro'
        (ej: 'classifier__max_depth').
    X_train : array-like
        Variables independientes de entrenamiento.
    y_train : array-like
        Variable objetivo de entrenamiento.
    method : {'grid', 'random'}, default 'grid'
        Método de búsqueda. 'grid' para GridSearchCV,
        'random' para RandomizedSearchCV.
    n_iter : int, default 10
        Número de combinaciones a probar (solo para method='random').
    cv : int, default 5
        Número de folds en la validación cruzada.
    scoring : str, default 'f1'
        Métrica a optimizar. Se usa F1 porque hay desbalance de clases.
    random_state : int, default 42
        Semilla para reproducibilidad.

    Retorna
    -------
    Pipeline
        Pipeline ajustado con los mejores hiperparámetros encontrados.
    """
    cv_strategy = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )

    if method == "grid":
        buscador = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )
    elif method == "random":
        buscador = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
        )
    else:
        raise ValueError(f"method debe ser 'grid' o 'random', recibido: '{method}'")

    buscador.fit(X_train, y_train)

    print(f"  ✅ Mejor {scoring}: {buscador.best_score_:.4f}")
    print(f"  🔧 Mejores parámetros: {buscador.best_params_}")

    return buscador.best_estimator_


# ──────────────────────────────────────────────
# Optimización de regresores
# ──────────────────────────────────────────────

def tune_regression_model(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "grid",
    n_iter: int = 10,
    cv: int = 5,
    scoring: str = "r2",
    random_state: int = 42,
) -> Pipeline:
    """
    Optimiza hiperparámetros de un pipeline de regresión.

    Usa GridSearchCV o RandomizedSearchCV con KFold estándar
    (apropiado para targets continuos).

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline base con el regresor.
    param_grid : dict
        Grilla de hiperparámetros. Convención: 'paso__parametro'
        (ej: 'regressor__max_depth').
    X_train : array-like
        Variables independientes de entrenamiento.
    y_train : array-like
        Variable objetivo continua de entrenamiento.
    method : {'grid', 'random'}, default 'grid'
        Método de búsqueda.
    n_iter : int, default 10
        Número de combinaciones (solo para method='random').
    cv : int, default 5
        Número de folds.
    scoring : str, default 'r2'
        Métrica a optimizar.
    random_state : int, default 42
        Semilla para reproducibilidad.

    Retorna
    -------
    Pipeline
        Pipeline ajustado con los mejores hiperparámetros encontrados.
    """
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    if method == "grid":
        buscador = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )
    elif method == "random":
        buscador = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
        )
    else:
        raise ValueError(f"method debe ser 'grid' o 'random', recibido: '{method}'")

    buscador.fit(X_train, y_train)

    print(f"  ✅ Mejor {scoring}: {buscador.best_score_:.4f}")
    print(f"  🔧 Mejores parámetros: {buscador.best_params_}")

    return buscador.best_estimator_


# ──────────────────────────────────────────────
# Serialización de modelos
# ──────────────────────────────────────────────

def save_model(
    modelo: Pipeline,
    nombre_archivo: str,
    save_dir: str = DEFAULT_MODELS_DIR,
) -> None:
    """
    Serializa un pipeline entrenado en disco usando joblib.

    Parámetros
    ----------
    modelo : Pipeline
        Pipeline ajustado a guardar.
    nombre_archivo : str
        Nombre del archivo de salida (ej: 'best_svm_classifier.joblib').
    save_dir : str
        Carpeta de destino. Se crea si no existe.
    """
    os.makedirs(save_dir, exist_ok=True)
    ruta = os.path.join(save_dir, nombre_archivo)
    joblib.dump(modelo, ruta)
    print(f"  💾 Modelo guardado en: {ruta}")


def load_model(
    nombre_archivo: str,
    save_dir: str = DEFAULT_MODELS_DIR,
) -> Pipeline:
    """
    Carga un pipeline serializado desde disco.

    Parámetros
    ----------
    nombre_archivo : str
        Nombre del archivo .joblib a cargar.
    save_dir : str
        Carpeta donde se encuentra el archivo.

    Retorna
    -------
    Pipeline
        Pipeline deserializado y listo para predecir.

    Lanza
    -----
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    """
    ruta = os.path.join(save_dir, nombre_archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el modelo en: '{ruta}'. "
            "Asegúrate de haber ejecutado el Notebook 4."
        )
    modelo = joblib.load(ruta)
    print(f"  ✅ Modelo cargado desde: {ruta}")
    return modelo
