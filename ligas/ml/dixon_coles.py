"""Modelo estadístico bivariante de Poisson (Dixon & Coles 1997) para fútbol.

Modela las tasas de goles esperados (lambda_local, mu_visitante) según la fuerza
de ataque y defensa de cada equipo, ventaja de campo y ajuste de baja anotación (rho).
Optimizado vectorialmente con NumPy para máxima velocidad de entrenamiento.
"""

import math
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


class DixonColesModel:
    """Implementación vectorizada de Dixon-Coles con estimación por máxima verosimilitud."""

    def __init__(self, rho: float = -0.05, max_goals: int = 7) -> None:
        self.rho = rho
        self.max_goals = max_goals
        self.team_attack: Dict[int, float] = {}
        self.team_defense: Dict[int, float] = {}
        self.home_advantage: float = 0.25
        self.intercept: float = 0.05
        self.teams: List[int] = []

    @staticmethod
    def _tau(x: int, y: int, lambda_param: float, mu_param: float, rho: float) -> float:
        """Factor de corrección de dependencia para resultados de baja anotación."""
        if x == 0 and y == 0:
            return 1.0 - lambda_param * mu_param * rho
        if x == 0 and y == 1:
            return 1.0 + lambda_param * rho
        if x == 1 and y == 0:
            return 1.0 + mu_param * rho
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def fit(self, partidos: List[Any], time_decay: float = 0.001) -> "DixonColesModel":
        """Ajusta los parámetros de ataque, defensa y ventaja de campo con optimización vectorizada."""
        partidos_jugados = [p for p in partidos if p.jugado]
        if len(partidos_jugados) < 15:
            self._fit_simple(partidos_jugados)
            return self

        teams_set = set()
        for p in partidos_jugados:
            teams_set.add(p.local_id)
            teams_set.add(p.visitante_id)
        self.teams = sorted(list(teams_set))
        n_teams = len(self.teams)
        team_idx = {t_id: i for i, t_id in enumerate(self.teams)}

        # Arrays vectorizados NumPy para cálculo ultra-rápido en C
        local_idx = np.array([team_idx[p.local_id] for p in partidos_jugados], dtype=int)
        visit_idx = np.array([team_idx[p.visitante_id] for p in partidos_jugados], dtype=int)
        x_arr = np.array([p.goles_local for p in partidos_jugados], dtype=float)
        y_arr = np.array([p.goles_visitante for p in partidos_jugados], dtype=float)

        # Máscaras booleanas precalculadas para casos de baja anotación
        m00 = (x_arr == 0) & (y_arr == 0)
        m01 = (x_arr == 0) & (y_arr == 1)
        m10 = (x_arr == 1) & (y_arr == 0)
        m11 = (x_arr == 1) & (y_arr == 1)

        def _neg_log_likelihood(params):
            home_adv = params[0]
            intercept = params[1]
            rho_val = params[2]
            attacks = params[3 : 3 + n_teams]
            defenses = params[3 + n_teams : 3 + 2 * n_teams]

            # Goles esperados lambda y mu (con límites numéricos estables)
            lam = np.exp(np.clip(intercept + home_adv + attacks[local_idx] - defenses[visit_idx], -4.0, 4.0))
            mu = np.exp(np.clip(intercept + attacks[visit_idx] - defenses[local_idx], -4.0, 4.0))

            # Factor tau vectorizado
            tau = np.ones_like(x_arr)
            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho_val
            tau[m01] = 1.0 + lam[m01] * rho_val
            tau[m10] = 1.0 + mu[m10] * rho_val
            tau[m11] = 1.0 - rho_val
            tau = np.maximum(tau, 1e-6)

            # Log-verosimilitud analítica vectorizada de Poisson
            log_lik = np.sum(np.log(tau) - lam + x_arr * np.log(lam) - mu + y_arr * np.log(mu))
            penalty = 100.0 * (np.mean(attacks) ** 2) + 10.0 * (np.mean(defenses) ** 2)

            return -log_lik + penalty

        init_params = np.zeros(3 + 2 * n_teams)
        init_params[0] = 0.25  # home advantage
        init_params[1] = 0.05  # intercept
        init_params[2] = -0.04  # rho

        bounds = [(-0.5, 1.0), (-1.0, 1.0), (-0.2, 0.2)] + [(-2.0, 2.0)] * (2 * n_teams)

        res = minimize(
            _neg_log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 80},
        )

        if res.success or res.x is not None:
            self.home_advantage = float(res.x[0])
            self.intercept = float(res.x[1])
            self.rho = float(res.x[2])
            attacks = res.x[3 : 3 + n_teams]
            defenses = res.x[3 + n_teams : 3 + 2 * n_teams]

            for idx, t_id in enumerate(self.teams):
                self.team_attack[t_id] = float(attacks[idx])
                self.team_defense[t_id] = float(defenses[idx])
        else:
            self._fit_simple(partidos_jugados)

        return self

    def _fit_simple(self, partidos: List[Any]) -> None:
        """Ajuste aproximado rápido cuando hay pocas muestras."""
        goles_f = {t: [] for t in self.teams}
        goles_c = {t: [] for t in self.teams}
        for p in partidos:
            goles_f.setdefault(p.local_id, []).append(p.goles_local)
            goles_c.setdefault(p.local_id, []).append(p.goles_visitante)
            goles_f.setdefault(p.visitante_id, []).append(p.goles_visitante)
            goles_c.setdefault(p.visitante_id, []).append(p.goles_local)

        avg_gf = 1.3
        for t_id in self.teams:
            gf = np.mean(goles_f[t_id]) if goles_f.get(t_id) else avg_gf
            gc = np.mean(goles_c[t_id]) if goles_c.get(t_id) else avg_gf
            self.team_attack[t_id] = math.log(max(gf / avg_gf, 0.5))
            self.team_defense[t_id] = -math.log(max(gc / avg_gf, 0.5))

    def expected_goals(self, local_id: int, visit_id: int) -> Tuple[float, float]:
        """Calcula lambda (goles esperados local) y mu (goles esperados visitante)."""
        att_l = self.team_attack.get(local_id, 0.0)
        def_l = self.team_defense.get(local_id, 0.0)
        att_v = self.team_attack.get(visit_id, 0.0)
        def_v = self.team_defense.get(visit_id, 0.0)

        lam = math.exp(self.intercept + self.home_advantage + att_l - def_v)
        mu = math.exp(self.intercept + att_v - def_l)

        # Clamping por estabilidad
        lam = min(max(lam, 0.2), 4.5)
        mu = min(max(mu, 0.2), 4.5)
        return lam, mu

    def score_probability_matrix(
        self, local_id: int, visit_id: int
    ) -> np.ndarray:
        """Devuelve una matriz (max_goals x max_goals) con P(Goles_Local=x, Goles_Visitante=y)."""
        lam, mu = self.expected_goals(local_id, visit_id)
        matrix = np.zeros((self.max_goals, self.max_goals))

        for x in range(self.max_goals):
            for y in range(self.max_goals):
                tau_val = self._tau(x, y, lam, mu, self.rho)
                p_x = poisson.pmf(x, lam)
                p_y = poisson.pmf(y, mu)
                matrix[x, y] = max(tau_val * p_x * p_y, 0.0)

        # Normalizar para que sume 1.0
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    def predict_1x2_probabilities(
        self, local_id: int, visit_id: int
    ) -> Tuple[float, float, float]:
        """Devuelve (p_local, p_empate, p_visitante)."""
        matrix = self.score_probability_matrix(local_id, visit_id)
        p_local = float(np.tril(matrix, -1).sum())  # x > y
        p_empate = float(np.trace(matrix))  # x == y
        p_visitante = float(np.triu(matrix, 1).sum())  # x < y

        total = p_local + p_empate + p_visitante
        if total > 0:
            return p_local / total, p_empate / total, p_visitante / total
        return 0.45, 0.28, 0.27

    def predict_pleno_al_15(
        self, local_id: int, visit_id: int
    ) -> Dict[str, Any]:
        """Calcula las probabilidades del Pleno al 15: categorías 0, 1, 2, M (3+ goles)

        para local y visitante, así como el marcador exacto más probable.
        """
        matrix = self.score_probability_matrix(local_id, visit_id)

        # Probabilidades marginales por equipo: 0, 1, 2, M (3 o más)
        def _get_categories(prob_vector):
            p0 = prob_vector[0]
            p1 = prob_vector[1]
            p2 = prob_vector[2]
            pm = sum(prob_vector[3:])
            return {"0": float(p0), "1": float(p1), "2": float(p2), "M": float(pm)}

        p_local_marginal = matrix.sum(axis=1)
        p_visit_marginal = matrix.sum(axis=0)

        cat_local = _get_categories(p_local_marginal)
        cat_visit = _get_categories(p_visit_marginal)

        # Marcador conjunto más probable
        max_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
        score_most_likely = (int(max_idx[0]), int(max_idx[1]))
        score_prob = float(matrix[max_idx])

        def _cat_score(goles):
            return "M" if goles >= 3 else str(goles)

        pleno_str = f"{_cat_score(score_most_likely[0])}-{_cat_score(score_most_likely[1])}"

        return {
            "cat_local": cat_local,
            "cat_visit": cat_visit,
            "marcador_probable": score_most_likely,
            "marcador_probable_prob": score_prob,
            "pleno_recomendado": pleno_str,
        }
