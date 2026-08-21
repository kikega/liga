"""Módulo de predicción: Modelo GBDT calibrado + Dixon-Coles persistido con joblib."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from django.conf import settings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit

from ligas.ml.dixon_coles import DixonColesModel
from ligas.ml.features import CLASE_A_RESULTADO, FEATURE_NAMES, FeatureExtractor
from ligas.models import Prediccion

MODEL_DIR = Path(settings.BASE_DIR) / "ml_models"
MODELO_DEFECTO = "ligas_predictor.joblib"


class Predictor:
    """Envoltura del modelo de clasificación 1/X/2 con calibración y Dixon-Coles."""

    def __init__(
        self,
        modelo=None,
        dixon_coles: Optional[DixonColesModel] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.modelo = modelo
        self.dixon_coles = dixon_coles or DixonColesModel()
        self.feature_names = feature_names or list(FEATURE_NAMES)

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> "Predictor":
        """Entrena un clasificador Gradient Boosting con calibración de probabilidades."""
        if len(y) >= 60:
            base_model = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=42,
            )
            # Calibración isotónica/sigmoide para probabilidades realistas
            cv_folds = min(3, max(2, len(y) // 30))
            self.modelo = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=cv_folds)
        else:
            # Fallback para muestras reducidas
            self.modelo = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

        self.modelo.fit(X, y)
        return self

    def predecir(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Devuelve (matriz de probabilidades 1/X/2 [N, 3], lista de resultados predichos)."""
        if self.modelo is None:
            raise ValueError("El modelo no está entrenado. Ejecuta train_model primero.")

        proba_raw = self.modelo.predict_proba(X)
        proba_orden = np.zeros((X.shape[0], 3))

        # Alinea las clases predichas con [0=1, 1=X, 2=2]
        classes = getattr(self.modelo, "classes_", [0, 1, 2])
        for col, clase in enumerate(classes):
            proba_orden[:, int(clase)] = proba_raw[:, col]

        # Asegura normalización a suma 1.0 por fila
        row_sums = proba_orden.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        proba_orden = proba_orden / row_sums

        pred_classes = np.argmax(proba_orden, axis=1)
        resultados = [CLASE_A_RESULTADO[int(c)] for c in pred_classes]
        return proba_orden, resultados

    def evaluar(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
        """Evalúa el modelo calculando Accuracy, Brier Score y reporte de clasificación."""
        proba, resultados = self.predecir(X)
        preds = np.argmax(proba, axis=1)

        acc = float(accuracy_score(y, preds))

        # Multi-class Brier Score = (1/N) * sum_i sum_c (p_ic - y_ic)^2
        y_one_hot = np.zeros_like(proba)
        for i, val in enumerate(y):
            y_one_hot[i, int(val)] = 1.0
        brier_score = float(np.mean(np.sum((proba - y_one_hot) ** 2, axis=1)))

        reporte = classification_report(y, preds, output_dict=True, zero_division=0)
        return acc, brier_score, reporte

    def guardar(self, path: Optional[str] = None) -> Path:
        target_path = Path(path) if path else MODEL_DIR / MODELO_DEFECTO
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "modelo": self.modelo,
                "dixon_coles": self.dixon_coles,
                "feature_names": self.feature_names,
            },
            target_path,
        )
        return target_path

    @classmethod
    def cargar(cls, path: Optional[str] = None) -> "Predictor":
        target_path = Path(path) if path else MODEL_DIR / MODELO_DEFECTO
        if not target_path.exists():
            raise FileNotFoundError(f"No existe el modelo entrenado en {target_path}")
        datos = joblib.load(target_path)
        return cls(
            modelo=datos.get("modelo"),
            dixon_coles=datos.get("dixon_coles"),
            feature_names=datos.get("feature_names"),
        )


def entrenar_modelo(
    partidos: List[Any],
    form_window: int = 5,
    output: Optional[str] = None,
    min_muestras: int = 30,
) -> Tuple[Predictor, float, float, Dict[str, Any], Path]:
    """Entrena, evalúa temporalmente y persiste el modelo predictivo con validación cronológica."""
    extractor = FeatureExtractor(form_window=form_window)
    datos = extractor.extract(partidos)

    if len(datos["y"]) < min_muestras:
        raise ValueError(
            f"Se necesitan al menos {min_muestras} partidos jugados para entrenar; hay {len(datos['y'])}."
        )

    X, y = datos["X"], datos["y"]
    n_samples = len(y)

    # Validación temporal (Train en 80% inicial, Test en 20% más reciente)
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Entrenar modelo Dixon-Coles
    dixon_coles = DixonColesModel().fit(partidos[:split_idx])

    # Entrenar clasificador GBDT
    predictor = Predictor(dixon_coles=dixon_coles, feature_names=datos["feature_names"]).entrenar(
        X_train, y_train
    )

    exactitud, brier, reporte = predictor.evaluar(X_test, y_test)

    # Re-entrenamiento final con el 100% de los datos para producción
    dixon_coles_full = DixonColesModel().fit(partidos)
    predictor_full = Predictor(
        dixon_coles=dixon_coles_full, feature_names=datos["feature_names"]
    ).entrenar(X, y)

    path = predictor_full.guardar(output)
    return predictor_full, exactitud, brier, reporte, path


def predecir_jornada(
    jornada: Any, path: Optional[str] = None, form_window: int = 5
) -> List[Prediccion]:
    """Predice todos los partidos de la jornada indicada y almacena las Prediccion."""
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
    extractor = FeatureExtractor(form_window=form_window)
    datos = extractor.extract(todos)

    filas = [datos["indices"][p.id] for p in partidos_a_predecir]
    X_pred = datos["X"][filas]

    predictor = Predictor.cargar(path)
    proba_gbdt, resultados = predictor.predecir(X_pred)

    predicciones = []
    for partido, probs_gbdt, resultado in zip(partidos_a_predecir, proba_gbdt, resultados):
        # Combinar con probabilidades de Dixon-Coles si está disponible
        try:
            p_dc = predictor.dixon_coles.predict_1x2_probabilities(
                partido.local_id, partido.visitante_id
            )
            # Ensamble suave: 70% GBDT + 30% Dixon-Coles
            p_l = 0.70 * probs_gbdt[0] + 0.30 * p_dc[0]
            p_e = 0.70 * probs_gbdt[1] + 0.30 * p_dc[1]
            p_v = 0.70 * probs_gbdt[2] + 0.30 * p_dc[2]
            sum_p = p_l + p_e + p_v
            p_l, p_e, p_v = p_l / sum_p, p_e / sum_p, p_v / sum_p
        except Exception:
            p_l, p_e, p_v = probs_gbdt[0], probs_gbdt[1], probs_gbdt[2]

        pred, _ = Prediccion.objects.update_or_create(
            partido=partido,
            defaults={
                "prob_local": float(p_l),
                "prob_empate": float(p_e),
                "prob_visitante": float(p_v),
                "resultado_predicho": resultado,
            },
        )
        predicciones.append(pred)

    return predicciones