"""Modelos de ML clásico y sus rejillas de hiperparámetros.

Todos son Pipelines de scikit-learn con estandarización (z = (x - media) / desv,
calculadas SOLO con los datos de entrenamiento de cada fold), para que las
características en escalas distintas (proporciones, contraste GLCM, L* 0-100)
pesen parecido en los modelos basados en distancias (KNN, SVM) y en la
regularización (regresión logística). Los árboles no la necesitan, pero no les
afecta.

El desbalance de "regular" (~3,8 % en train) se compensa con pesos de clase
inversamente proporcionales a la frecuencia: w_c = n / (K * n_c). Así, en la
función de pérdida, equivocarse en una imagen "regular" cuesta ~9 veces más que
en una "good".
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import SEED


def _pipe(modelo) -> Pipeline:
    return Pipeline([("escala", StandardScaler()), ("modelo", modelo)])


def lineas_base() -> dict:
    """Clasificadores triviales: sirven para saber si un modelo "aprendió algo"."""
    return {
        "Dummy (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Dummy (aleatorio estratificado)": DummyClassifier(strategy="stratified", random_state=SEED),
    }


def modelos_y_rejillas(pesos_clase: bool = True) -> dict:
    """Devuelve {nombre: (pipeline, rejilla)} para GridSearchCV.

    XGBoost no acepta class_weight: sus pesos se pasan como sample_weight en fit.
    """
    cw = "balanced" if pesos_clase else None
    return {
        "Regresión logística": (
            _pipe(LogisticRegression(class_weight=cw, max_iter=5000, random_state=SEED)),
            {"modelo__C": [0.01, 0.1, 1, 10]},
        ),
        "KNN": (
            _pipe(KNeighborsClassifier()),
            {"modelo__n_neighbors": [5, 11, 21, 41], "modelo__weights": ["uniform", "distance"],
             "modelo__p": [1, 2]},
        ),
        "SVM (RBF)": (
            # probability=False en la búsqueda (Platt scaling hace un CV interno: x5 tiempo);
            # se activa al reentrenar el modelo final si se necesitan probabilidades (app)
            _pipe(SVC(kernel="rbf", class_weight=cw, probability=False, random_state=SEED)),
            {"modelo__C": [0.3, 1, 3, 10, 30], "modelo__gamma": ["scale", 0.003, 0.01, 0.03]},
        ),
        "Random Forest": (
            _pipe(RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample" if pesos_clase else None,
                                         random_state=SEED, n_jobs=-1)),
            {"modelo__max_depth": [None, 12, 20], "modelo__min_samples_leaf": [1, 3, 8],
             "modelo__max_features": ["sqrt", 0.3]},
        ),
        "XGBoost": (
            _pipe(XGBClassifier(n_estimators=400, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                eval_metric="mlogloss", random_state=SEED, n_jobs=-1)),
            {"modelo__max_depth": [3, 5, 7], "modelo__learning_rate": [0.03, 0.1],
             "modelo__min_child_weight": [1, 5]},
        ),
    }
