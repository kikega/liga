"""Ingeniería de características para predecir el resultado de un partido (1/X/2).

Características construidas por partido (con garantía estricta de CERO fuga temporal):

- Ratings Elo: Elo local, Elo visitante, diferencia de Elo y probabilidad esperada Elo.
- Forma reciente general: puntos y diferencia de goles en los últimos N partidos.
- Forma específica de campo: puntos y DG del local en casa / puntos y DG del visitante fuera.
- Rendimiento acumulado de temporada: puntos, goles a favor y en contra por partido.
- Días de descanso / fatiga entre encuentros.
- Impacto de ausencias: peso diferencial de los jugadores clave ausentes.
- Rendimiento sin jugadores clave evaluado únicamente con partidos previos.
- Head-to-head histórico entre ambos equipos previo a la fecha del partido.
- Factor campo (home_bias).
"""

from datetime import datetime
from typing import Any, Dict, List
import numpy as np

from ligas.ml.elo import EloCalculator
from ligas.models import (
    PUNTOS_EMPATE,
    PUNTOS_VICTORIA,
    RESULTADO_EMPATE,
    RESULTADO_LOCAL,
    RESULTADO_VISITANTE,
)

# Orden fijo de las características. Debe coincidir entre entrenamiento e inferencia.
FEATURE_NAMES = [
    "local_elo",
    "visit_elo",
    "elo_diff",
    "elo_exp_local",
    "local_form_points",
    "local_form_gd",
    "visit_form_points",
    "visit_form_gd",
    "local_home_form_pts",
    "local_home_form_gd",
    "visit_away_form_pts",
    "visit_away_form_gd",
    "local_season_pts_pm",
    "local_season_gf_pm",
    "local_season_gc_pm",
    "visit_season_pts_pm",
    "visit_season_gf_pm",
    "visit_season_gc_pm",
    "local_rest_days",
    "visit_rest_days",
    "local_absence_weight",
    "visit_absence_weight",
    "local_without_key_rate",
    "visit_without_key_rate",
    "h2h_local_wins",
    "h2h_draws",
    "h2h_away_wins",
    "home_bias",
]

RESULTADO_A_CLASE = {RESULTADO_LOCAL: 0, RESULTADO_EMPATE: 1, RESULTADO_VISITANTE: 2}
CLASE_A_RESULTADO = {v: k for k, v in RESULTADO_A_CLASE.items()}


class FeatureExtractor:
    """Construye la matriz de características a partir de los partidos de la BDD.

    Los partidos deben llegar con ``select_related("jornada__temporada", "local",
    "visitante")`` y ``prefetch_related("ausencias__jugador")``.
    """

    def __init__(
        self,
        form_window: int = 5,
        home_bias: float = 1.0,
        peso_jugador_clave: float = 1.0,
    ) -> None:
        self.form_window = form_window
        self.home_bias = home_bias
        self.peso_jugador_clave = peso_jugador_clave
        self.elo_calculator = EloCalculator()

    def extract(self, partidos: List[Any]) -> Dict[str, Any]:
        """Devuelve dict con X, y, feature_names e índices partido.id -> fila.

        ``y`` solo incluye las etiquetas de los partidos jugados; ``indices``
        mapea cada partido (jugado o no) a su fila en X.
        """
        partidos_ordenados = sorted(
            partidos,
            key=lambda p: (
                getattr(p.jornada.temporada, "inicio", None) or p.jornada.temporada_id,
                p.jornada.numero,
                p.fecha or datetime.min,
                p.id,
            ),
        )

        elo_map = self.elo_calculator.compute_match_ratings(partidos_ordenados)

        # Estructuras incrementales para evitar fuga de datos
        historial_acumulado: Dict[int, List[Dict[str, Any]]] = {}
        filas = []
        etiquetas = []
        indices = {}

        for p in partidos_ordenados:
            indices[p.id] = len(filas)
            features = self._features_para_partido(p, historial_acumulado, elo_map)
            filas.append(features)

            if p.resultado is not None:
                etiquetas.append(RESULTADO_A_CLASE[p.resultado])

            # Solo después de extraer features del partido p, se añade su resultado al historial
            if p.jugado:
                self._actualizar_historial(p, historial_acumulado)

        return {
            "X": np.array(filas, dtype=float),
            "y": np.array(etiquetas, dtype=int) if etiquetas else np.array([], dtype=int),
            "feature_names": list(FEATURE_NAMES),
            "indices": indices,
        }

    def _actualizar_historial(
        self, p: Any, historial: Dict[int, List[Dict[str, Any]]]
    ) -> None:
        """Registra el partido jugado en el historial acumulado."""
        ausencias = list(p.ausencias.all()) if hasattr(p, "ausencias") else []
        ausentes_local = {
            a.jugador_id
            for a in ausencias
            if a.jugador.es_importante and a.jugador.activo and a.jugador.equipo_id == p.local_id
        }
        ausentes_visitante = {
            a.jugador_id
            for a in ausencias
            if a.jugador.es_importante and a.jugador.activo and a.jugador.equipo_id == p.visitante_id
        }

        # Registro para equipo local
        historial.setdefault(p.local_id, []).append(
            {
                "partido_id": p.id,
                "fecha": p.fecha,
                "es_local": True,
                "gf": p.goles_local,
                "ga": p.goles_visitante,
                "puntos": p.puntos_local(),
                "rival_id": p.visitante_id,
                "ausentes_clave": ausentes_local,
                "temporada_id": p.jornada.temporada_id,
            }
        )

        # Registro para equipo visitante
        historial.setdefault(p.visitante_id, []).append(
            {
                "partido_id": p.id,
                "fecha": p.fecha,
                "es_local": False,
                "gf": p.goles_visitante,
                "ga": p.goles_local,
                "puntos": p.puntos_visitante(),
                "rival_id": p.local_id,
                "ausentes_clave": ausentes_visitante,
                "temporada_id": p.jornada.temporada_id,
            }
        )

    def _features_para_partido(
        self,
        p: Any,
        historial: Dict[int, List[Dict[str, Any]]],
        elo_map: Dict[int, Dict[str, float]],
    ) -> List[float]:
        """Calcula el vector de características exactas para el partido."""
        h_local = historial.get(p.local_id, [])
        h_visitante = historial.get(p.visitante_id, [])

        # 1. Elo Ratings
        elo_data = elo_map.get(
            p.id,
            {
                "local_elo": 1500.0,
                "visit_elo": 1500.0,
                "elo_diff": 0.0,
                "elo_exp_local": 0.5,
            },
        )

        # 2. Forma reciente general
        local_form = self._forma(h_local)
        visit_form = self._forma(h_visitante)

        # 3. Forma específica (local en casa / visitante fuera)
        h_local_casa = [m for m in h_local if m["es_local"]]
        h_visit_fuera = [m for m in h_visitante if not m["es_local"]]
        local_home_form = self._forma(h_local_casa)
        visit_away_form = self._forma(h_visit_fuera)

        # 4. Rendimiento acumulado en temporada actual
        temp_id = p.jornada.temporada_id
        h_local_temp = [m for m in h_local if m.get("temporada_id") == temp_id]
        h_visit_temp = [m for m in h_visitante if m.get("temporada_id") == temp_id]

        # 5. Días de descanso
        local_rest = self._dias_descanso(p, h_local)
        visit_rest = self._dias_descanso(p, h_visitante)

        # 6. Ausencias de jugadores clave
        local_abs = self._ausentes_importantes(p, p.local_id)
        visit_abs = self._ausentes_importantes(p, p.visitante_id)

        # 7. Eficiencia histórica sin jugadores clave
        local_without = self._sin_clave_rate(p.local_id, local_abs, h_local)
        visit_without = self._sin_clave_rate(p.visitante_id, visit_abs, h_visitante)

        # 8. Head-to-Head histórico
        h2h_stats = self._calcular_h2h(p.local_id, p.visitante_id, h_local)

        return [
            elo_data["local_elo"],
            elo_data["visit_elo"],
            elo_data["elo_diff"],
            elo_data["elo_exp_local"],
            local_form["points"],
            local_form["goal_diff"],
            visit_form["points"],
            visit_form["goal_diff"],
            local_home_form["points"],
            local_home_form["goal_diff"],
            visit_away_form["points"],
            visit_away_form["goal_diff"],
            self._acumulado(h_local_temp, "pts_pm"),
            self._acumulado(h_local_temp, "gf_pm"),
            self._acumulado(h_local_temp, "gc_pm"),
            self._acumulado(h_visit_temp, "pts_pm"),
            self._acumulado(h_visit_temp, "gf_pm"),
            self._acumulado(h_visit_temp, "gc_pm"),
            local_rest,
            visit_rest,
            len(local_abs) * self.peso_jugador_clave,
            len(visit_abs) * self.peso_jugador_clave,
            local_without,
            visit_without,
            h2h_stats[0],
            h2h_stats[1],
            h2h_stats[2],
            self.home_bias,
        ]

    def _forma(self, historial_equipo: List[Dict[str, Any]]) -> Dict[str, float]:
        """Puntos y diferencia de goles en los últimos N partidos de la lista."""
        recientes = historial_equipo[-self.form_window :]
        if not recientes:
            return {"points": 0.0, "goal_diff": 0.0}
        return {
            "points": float(sum(m["puntos"] for m in recientes)),
            "goal_diff": float(sum((m["gf"] - m["ga"]) for m in recientes)),
        }

    def _acumulado(self, historial_equipo: List[Dict[str, Any]], campo: str) -> float:
        """Promedios acumulados hasta el momento."""
        n = len(historial_equipo)
        if n == 0:
            return 0.0
        if campo == "pts_pm":
            return float(sum(m["puntos"] for m in historial_equipo) / n)
        if campo == "gf_pm":
            return float(sum(m["gf"] for m in historial_equipo) / n)
        return float(sum(m["ga"] for m in historial_equipo) / n)

    def _dias_descanso(self, partido_actual: Any, historial_equipo: List[Dict[str, Any]]) -> float:
        """Días de descanso desde el partido anterior (por defecto 7.0 si no hay registro)."""
        if not historial_equipo or not partido_actual.fecha:
            return 7.0
        ultimo_partido = historial_equipo[-1]
        fecha_ant = ultimo_partido.get("fecha")
        if not fecha_ant:
            return 7.0
        try:
            delta = (partido_actual.fecha - fecha_ant).total_seconds() / 86400.0
            return float(min(max(delta, 1.0), 30.0))
        except Exception:
            return 7.0

    def _ausentes_importantes(self, p: Any, equipo_id: int) -> set:
        if not hasattr(p, "ausencias"):
            return set()
        return {
            a.jugador_id
            for a in p.ausencias.all()
            if a.jugador.es_importante and a.jugador.activo and a.jugador.equipo_id == equipo_id
        }

    def _sin_clave_rate(
        self, equipo_id: int, ausentes: set, historial_equipo: List[Dict[str, Any]]
    ) -> float:
        """Eficiencia media de puntos previa en partidos donde faltaron estos jugadores."""
        if not ausentes or not historial_equipo:
            if not historial_equipo:
                return 0.5
            total_pts = sum(m["puntos"] for m in historial_equipo)
            return float(total_pts / (PUNTOS_VICTORIA * len(historial_equipo)))

        partidos_sin = [
            m for m in historial_equipo if any(j_id in m["ausentes_clave"] for j_id in ausentes)
        ]
        if not partidos_sin:
            total_pts = sum(m["puntos"] for m in historial_equipo)
            return float(total_pts / (PUNTOS_VICTORIA * len(historial_equipo)))

        pts_sin = sum(m["puntos"] for m in partidos_sin)
        return float(pts_sin / (PUNTOS_VICTORIA * len(partidos_sin)))

    def _calcular_h2h(
        self, local_id: int, rival_id: int, historial_local: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """Victorias, empates y derrotas del equipo local en sus enfrentamientos previos con el rival."""
        victorias = 0.0
        empates = 0.0
        derrotas = 0.0

        for m in historial_local:
            if m["rival_id"] == rival_id:
                if m["puntos"] == PUNTOS_VICTORIA:
                    victorias += 1.0
                elif m["puntos"] == PUNTOS_EMPATE:
                    empates += 1.0
                else:
                    derrotas += 1.0

        return victorias, empates, derrotas