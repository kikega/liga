"""Sistema de puntuación Elo dinámico para equipos de fútbol.

Calcula la fuerza relativa de cada equipo partido a partido, incorporando:
- Ventaja de campo (Home Advantage).
- Ponderación por diferencia de goles (Goal Margin Multiplier).
- Ajuste entre divisiones (Primera vs Segunda).
- Extracción de ratings históricos previos al partido (sin fuga temporal).
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple


class EloCalculator:
    """Calculadora y rastreador de ratings Elo para competiciones de fútbol."""

    def __init__(
        self,
        k_factor: float = 24.0,
        home_advantage: float = 65.0,
        initial_rating_div1: float = 1500.0,
        initial_rating_div2: float = 1350.0,
    ) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating_div1 = initial_rating_div1
        self.initial_rating_div2 = initial_rating_div2

    def initial_rating(self, division_nivel: int | None = 1) -> float:
        """Devuelve el rating base según la división."""
        if division_nivel == 2:
            return self.initial_rating_div2
        return self.initial_rating_div1

    def expected_score(self, rating_local: float, rating_visitante: float) -> Tuple[float, float]:
        """Calcula la puntuación esperada (probabilidad de victoria/empate ajustada)

        según la diferencia de Elo incluyendo la ventaja de campo.
        """
        diff = (rating_local + self.home_advantage) - rating_visitante
        exp_local = 1.0 / (1.0 + math.pow(10.0, -diff / 400.0))
        exp_visitante = 1.0 - exp_local
        return exp_local, exp_visitante

    def goal_margin_multiplier(self, goal_diff: int, elo_diff: float) -> float:
        """Multiplicador de margen de victoria según la fórmula oficial de World Football Elo."""
        abs_diff = abs(goal_diff)
        if abs_diff <= 1:
            return 1.0
        if abs_diff == 2:
            return 1.5
        return (11.0 + abs_diff) / 8.0

    def calculate_update(
        self,
        rating_local: float,
        rating_visitante: float,
        goles_local: int,
        goles_visitante: int,
    ) -> Tuple[float, float]:
        """Calcula el cambio de puntos Elo tras el partido."""
        exp_local, exp_visitante = self.expected_score(rating_local, rating_visitante)

        if goles_local > goles_visitante:
            actual_local = 1.0
            actual_visitante = 0.0
        elif goles_local == goles_visitante:
            actual_local = 0.5
            actual_visitante = 0.5
        else:
            actual_local = 0.0
            actual_visitante = 1.0

        goal_diff = goles_local - goles_visitante
        elo_diff = rating_local - rating_visitante
        mult = self.goal_margin_multiplier(goal_diff, elo_diff)

        delta_local = self.k_factor * mult * (actual_local - exp_local)
        delta_visitante = self.k_factor * mult * (actual_visitante - exp_visitante)

        return delta_local, delta_visitante

    def compute_match_ratings(
        self, partidos_ordenados: List[Any]
    ) -> Dict[int, Dict[str, float]]:
        """Procesa una secuencia cronológica de partidos y devuelve para cada partido

        el rating previo de ambos equipos y el rating diferencial.
        Garantiza CERO fuga de datos: el rating del partido t refleja solo partidos < t.
        """
        current_ratings: Dict[int, float] = {}
        history_map: Dict[int, Dict[str, float]] = {}

        for p in partidos_ordenados:
            local_id = p.local_id
            visit_id = p.visitante_id

            div_local = getattr(p, "division_nivel", 1) or 1
            if local_id not in current_ratings:
                current_ratings[local_id] = self.initial_rating(div_local)
            if visit_id not in current_ratings:
                current_ratings[visit_id] = self.initial_rating(div_local)

            r_local = current_ratings[local_id]
            r_visit = current_ratings[visit_id]
            elo_diff = (r_local + self.home_advantage) - r_visit
            exp_local, exp_visit = self.expected_score(r_local, r_visit)

            history_map[p.id] = {
                "local_elo": r_local,
                "visit_elo": r_visit,
                "elo_diff": elo_diff,
                "elo_exp_local": exp_local,
                "elo_exp_visit": exp_visit,
            }

            # Actualiza el rating solo si el partido fue jugado
            if p.jugado:
                d_local, d_visit = self.calculate_update(
                    r_local, r_visit, p.goles_local, p.goles_visitante
                )
                current_ratings[local_id] = r_local + d_local
                current_ratings[visit_id] = r_visit + d_visit

        return history_map
