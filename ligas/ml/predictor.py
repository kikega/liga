"""Módulo de predicción: modelo RandomForest persistido con joblib."""

from pathlib import Path

import joblib
import numpy as np
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ligas.ml.features import CLASE_A_RESULTADO, FEATURE_NAMES, FeatureExtractor
from ligas.models import Prediccion

MODEL_DIR = Path(settings.BASE_DIR) / "ml_models"
MODELO_DEFECTO = "ligas_predictor.joblib"


class Predictor:
    """Envoltura del modelo de clasificación 1/X/2 con persistencia joblib."""

    def __init__(self, modelo=None, feature_names=None):
        self.modelo = modelo
        self.feature_names = feature_names or list(FEATURE_NAMES)

    def entrenar(self, X, y):
        self.modelo = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.modelo.fit(X, y)
        return self

    def predecir(self, X):
        """Devuelve (probabilidades 1/X/2, lista de resultados) para cada fila."""
        if self.modelo is None:
            raise ValueError("El modelo no está entrenado. Ejecuta train_model primero.")
        proba = self.modelo.predict_proba(X)
        proba_orden = np.zeros((X.shape[0], 3))
        for col, clase in enumerate(self.modelo.classes_):
            proba_orden[:, int(clase)] = proba[:, col]
        resultados = [CLASE_A_RESULTADO[int(c)] for c in self.modelo.predict(X)]
        return proba_orden, resultados

    def evaluar(self, X, y):
        predicciones = self.modelo.predict(X)
        return (
            accuracy_score(y, predicciones),
            classification_report(y, predicciones, output_dict=True, zero_division=0),
        )

    def guardar(self, path=None):
        path = Path(path) if path else MODEL_DIR / MODELO_DEFECTO
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modelo": self.modelo, "feature_names": self.feature_names}, path)
        return path

    @classmethod
    def cargar(cls, path=None):
        path = Path(path) if path else MODEL_DIR / MODELO_DEFECTO
        if not path.exists():
            raise FileNotFoundError(f"No existe el modelo entrenado en {path}")
        datos = joblib.load(path)
        return cls(modelo=datos["modelo"], feature_names=datos["feature_names"])


def entrenar_modelo(partidos, form_window=5, output=None, min_muestras=30):
    """Entrena, evalúa y persiste el modelo a partir de una lista de Partido."""
    datos = FeatureExtractor(form_window=form_window).extract(partidos)
    if len(datos["y"]) < min_muestras:
        raise ValueError(
            f"Se necesitan al menos {min_muestras} partidos jugados para entrenar; hay {len(datos['y'])}."
        )
    X, y = datos["X"], datos["y"]
    stratify = y if all((y == c).sum() >= 2 for c in np.unique(y)) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    predictor = Predictor(feature_names=datos["feature_names"]).entrenar(X_train, y_train)
    exactitud, reporte = predictor.evaluar(X_test, y_test)
    path = predictor.guardar(output)
    return predictor, exactitud, reporte, path


def predecir_jornada(jornada, path=None, form_window=5):
    """Predice la jornada, persiste las Prediccion y las devuelve.

    Usa el histórico completo de la temporada para construir las features de
    los partidos pendientes de la jornada indicada.
    """
    partidos_a_predecir = list(
        jornada.partidos.select_related("local", "visitante").prefetch_related("ausencias__jugador")
    )
    if not partidos_a_predecir:
        return []

    historico = list(
        jornada.temporada.partidos_historicos()
        .select_related("jornada__temporada", "local", "visitante")
        .prefetch_related("ausencias__jugador")
    )
    todos = historico + partidos_a_predecir
    datos = FeatureExtractor(form_window=form_window).extract(todos)

    filas = [datos["indices"][p.id] for p in partidos_a_predecir]
    X = datos["X"][filas]

    predictor = Predictor.cargar(path)
    proba, resultados = predictor.predecir(X)

    predicciones = []
    for partido, probs, resultado in zip(partidos_a_predecir, proba, resultados):
        pred, _ = Prediccion.objects.update_or_create(
            partido=partido,
            defaults={
                "prob_local": float(probs[0]),
                "prob_empate": float(probs[1]),
                "prob_visitante": float(probs[2]),
                "resultado_predicho": resultado,
            },
        )
        predicciones.append(pred)
    return predicciones