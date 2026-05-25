"""
data_preprocessing.py
=====================
Funciones de limpieza, transformación y preparación de datos
para el proyecto de predicción de abandono de clientes.

Asignatura : Programación para la Ciencia de Datos (SCY1101)
Evaluación  : Evaluación Parcial N°2
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ──────────────────────────────────────────────
# Clases transformer personalizadas
# ──────────────────────────────────────────────

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Recorta los valores extremos de cada columna numérica al percentil
    inferior y superior indicado en `limits`.

    Parámetros
    ----------
    limits : tuple, default (0.05, 0.05)
        Fracción a recortar en cada cola (inferior, superior).

    Atributos
    ---------
    columns_ : array-like
        Nombres de las columnas procesadas.
    """

    def __init__(self, limits=(0.05, 0.05)):
        self.limits = limits

    def fit(self, X, y=None):
        self.columns_ = (
            X.columns if isinstance(X, pd.DataFrame) else np.arange(X.shape[1])
        )
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        for col in self.columns_:
            lower = X[col].quantile(self.limits[0])
            upper = X[col].quantile(1 - self.limits[1])
            X[col] = np.clip(X[col].astype("float64"), lower, upper)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(
            self.columns_ if input_features is None else input_features
        )


# ──────────────────────────────────────────────
# Funciones de limpieza sobre el DataFrame crudo
# ──────────────────────────────────────────────

def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina registros duplicados basándose en la columna 'id_cliente'.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset original.

    Retorna
    -------
    pd.DataFrame
        Dataset sin duplicados por 'id_cliente'.
    """
    antes = len(df)
    df = df.drop_duplicates(subset=["id_cliente"]).copy()
    print(f"  Duplicados eliminados: {antes - len(df)}")
    return df


def tratar_negativos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte a NaN los valores negativos en columnas que por naturaleza
    no pueden serlo (ingreso_mensual, gasto_mensual, deuda_total).

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset de entrada.

    Retorna
    -------
    pd.DataFrame
        Dataset con negativos reemplazados por NaN.
    """
    columnas = ["ingreso_mensual", "gasto_mensual", "deuda_total"]
    df = df.copy()
    for col in columnas:
        n = (df[col] < 0).sum()
        if n > 0:
            print(f"  '{col}': {n} valores negativos convertidos a NaN")
        df.loc[df[col] < 0, col] = np.nan
    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera nuevas variables a partir de las existentes:
      - ratio_deuda : deuda_total / (ingreso_mensual + 1)
      - hora_sin    : seno cíclico de hora_registro (24 h)
      - hora_cos    : coseno cíclico de hora_registro (24 h)

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset limpio (sin negativos).

    Retorna
    -------
    pd.DataFrame
        Dataset con las nuevas columnas añadidas.
    """
    df = df.copy()
    df["ratio_deuda"] = df["deuda_total"] / (df["ingreso_mensual"] + 1)
    df["hora_sin"] = np.sin(2 * np.pi * df["hora_registro"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora_registro"] / 24)
    return df


# ──────────────────────────────────────────────
# Definición de columnas por tipo
# ──────────────────────────────────────────────

# Columnas que NO entran al modelo
COLS_EXCLUIR = ["id_cliente", "fecha_registro", "codigo_postal"]

# Variables numéricas que pasan por el pipeline
NUMERIC_FEATURES = [
    "edad", "ingreso_mensual", "antiguedad_meses", "frecuencia_compra",
    "ultima_compra_dias", "num_productos", "hora_registro",
    "gasto_mensual", "deuda_total", "score_crediticio",
    "ratio_deuda", "hora_sin", "hora_cos",
]

# Variables categóricas que pasan por el pipeline
CATEGORICAL_FEATURES = [
    "genero", "estado_civil", "canal_registro", "dia_semana_registro",
    "region", "tipo_plan", "uso_app",
]

# Variable binaria tratada como numérica (ya es 0/1)
BINARY_FEATURES = ["tiene_tarjeta_credito"]


# ──────────────────────────────────────────────
# Construcción del ColumnTransformer
# ──────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Construye el ColumnTransformer con los pipelines numérico y categórico.

    Pipeline numérico  : Winsorizer → SimpleImputer(mean) → StandardScaler
    Pipeline categórico: SimpleImputer(most_frequent) → OneHotEncoder

    Retorna
    -------
    ColumnTransformer
        Preprocesador listo para usar dentro de un Pipeline de Scikit-learn.
    """
    numeric_transformer = Pipeline(steps=[
        ("winsorizer", Winsorizer()),
        ("imputer",    SimpleImputer(strategy="mean")),
        ("escalado",   StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES + BINARY_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


# ──────────────────────────────────────────────
# Pipeline completo de preprocesamiento
# ──────────────────────────────────────────────

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la secuencia completa de limpieza y feature engineering al
    dataset crudo y retorna un DataFrame con todas las variables listas
    para el modelado (columnas numéricas y dummies).

    Pasos:
      1. Eliminar duplicados por id_cliente.
      2. Tratar valores negativos → NaN.
      3. Crear variables derivadas (ratio_deuda, variables cíclicas).
      4. Eliminar columnas no informativas.
      5. Aplicar el ColumnTransformer (Winsorizer + imputer + scaler + OHE).

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset original crudo.

    Retorna
    -------
    pd.DataFrame
        Dataset limpio y transformado, listo para train/test split.
    """
    print("🔧 Iniciando preprocesamiento...")

    # Pasos manuales sobre el DataFrame
    df = eliminar_duplicados(df)
    df = tratar_negativos(df)
    df = crear_features(df)

    # Separar target antes de transformar
    y = df["abandono"].reset_index(drop=True)

    # Eliminar columnas que no aportan al modelo
    cols_drop = COLS_EXCLUIR + ["abandono"]
    X_raw = df.drop(columns=[c for c in cols_drop if c in df.columns])

    # Aplicar preprocesador
    preprocessor = build_preprocessor()
    X_array = preprocessor.fit_transform(X_raw)

    # Reconstruir nombres de columnas
    ohe_cols = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )
    all_cols = NUMERIC_FEATURES + BINARY_FEATURES + ohe_cols

    df_clean = pd.DataFrame(X_array, columns=all_cols)
    df_clean["abandono"] = y.values

    print(f"✅ Preprocesamiento finalizado. Shape: {df_clean.shape}")
    return df_clean
