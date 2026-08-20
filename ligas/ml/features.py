"""Ingeniería de características para predecir el resultado de un partido (1/X/2).

Características construidas por partido:

- Forma reciente: puntos y diferencia de goles en los últimos N partidos.
- Rendimiento acumulado de temporada: puntos, goles a favor y en contra por partido.
- Impacto de ausencias: peso diferencial de los jugadores clave ausentes.
- Rendimiento sin jugadores clave: eficiencia de puntos del equipo en partidos
  históricos donde faltó cada jugador clave.
- Head-to-head histórico entre ambos equipos.
- Factor campo: feature explícita ``home_bias`` además de las features
  asimétricas local/visitante.
"""

import numpy as np

from ligas.models import (
    PUNTOS_EMPATE,
    PUNTOS_VICTORIA,
    RESULTADO_EMPATE,
    RESULTADO_LOCAL,
    RESULTADO_VISITANTE,
)

# Orden fijo de las características. Debe coincidir entre entrenamiento e inferencia.
FEATURE_NAMES = [
    "local_form_points",
    "local_form_gd",
    "visit_form_points",
    "visit_form_gd",
    "local_season_pts_pm",
    "local_season_gf_pm",
    "local_season_gc_pm",
    "visit_season_pts_pm",
    "visit_season_gf_pm",
    "visit_season_gc_pm",
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

    def __init__(self, form_window=5, home_bias=1.0, peso_jugador_clave=1.0):
        self.form_window = form_window
        self.home_bias = home_bias
        self.peso_jugador_clave = peso_jugador_clave

    def extract(self, partidos):
        """Devuelve dict con X, y, feature_names e índices partido.id -> fila.

        ``y`` solo incluye las etiquetas de los partidos jugados; ``indices``
        mapea cada partido (jugado o no) a su fila en X.
        """
        partidos = sorted(
            partidos,
            key=lambda p: (p.jornada.temporada_id, p.jornada.numero, p.id),
        )

        historial = self._construir_historial(partidos)
        sin_clave = self._rendimiento_sin_clave(historial)
        h2h = self._head_to_head(historial)

        filas = []
        etiquetas = []
        indices = {}
        for p in partidos:
            indices[p.id] = len(filas)
            filas.append(self._features_para(p, historial, sin_clave, h2h))
            if p.resultado is not None:
                etiquetas.append(RESULTADO_A_CLASE[p.resultado])

        return {
            "X": np.array(filas, dtype=float),
            "y": np.array(etiquetas, dtype=int) if etiquetas else np.array([], dtype=int),
            "feature_names": list(FEATURE_NAMES),
            "indices": indices,
        }

    def _construir_historial(self, partidos):
        """Historial por equipo con todos los partidos jugados (orden cronológico)."""
        historial = {}
        for p in partidos:
            if not p.jugado:
                continue
            ausencias = list(p.ausencias.all())
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
            for equipo_id, gf, ga, puntos, rival_id, ausentes in (
                (p.local_id, p.goles_local, p.goles_visitante, p.puntos_local(), p.visitante_id, ausentes_local),
                (p.visitante_id, p.goles_visitante, p.goles_local, p.puntos_visitante(), p.local_id, ausentes_visitante),
            ):
                historial.setdefault(equipo_id, []).append(
                    {
                        "partido_id": p.id,
                        "gf": gf,
                        "ga": ga,
                        "puntos": puntos,
                        "rival_id": rival_id,
                        "ausentes_clave": ausentes,
                    }
                )
        return historial

    def _rendimiento_sin_clave(self, historial):
        """Eficiencia de puntos (puntos/3) del equipo cuando falta cada jugador clave."""
        ausencias = {}
        for equipo_id, partidos_equipo in historial.items():
            for m in partidos_equipo:
                for jugador_id in m["ausentes_clave"]:
                    ausencias.setdefault(equipo_id, {}).setdefault(jugador_id, []).append(m["puntos"])

        por_jugador = {}
        for equipo_id, jugadores in ausencias.items():
            for jugador_id, puntos_lista in jugadores.items():
                por_jugador[(equipo_id, jugador_id)] = sum(puntos_lista) / (PUNTOS_VICTORIA * len(puntos_lista))

        global_team = {}
        for equipo_id, partidos_equipo in historial.items():
            total_puntos = sum(m["puntos"] for m in partidos_equipo)
            global_team[equipo_id] = total_puntos / (PUNTOS_VICTORIA * len(partidos_equipo))

        return {"por_jugador": por_jugador, "global": global_team}

    def _head_to_head(self, historial):
        """Cuenta victorias/empates entre pares (equipo_local, equipo_visitante)."""
        h2h = {}
        for equipo_id, partidos_equipo in historial.items():
            for m in partidos_equipo:
                clave = (equipo_id, m["rival_id"])
                triple = h2h.setdefault(clave, [0, 0, 0])
                if m["puntos"] == PUNTOS_VICTORIA:
                    triple[0] += 1
                elif m["puntos"] == PUNTOS_EMPATE:
                    triple[1] += 1
                else:
                    triple[2] += 1
        return h2h

    def _features_para(self, p, historial, sin_clave, h2h):
        h_local = [m for m in historial.get(p.local_id, []) if m["partido_id"] != p.id]
        h_visitante = [m for m in historial.get(p.visitante_id, []) if m["partido_id"] != p.id]

        local_form = self._forma(h_local)
        visit_form = self._forma(h_visitante)

        local_abs = self._ausentes_importantes(p, p.local_id)
        visit_abs = self._ausentes_importantes(p, p.visitante_id)

        local_without = self._sin_clave_rate(p.local_id, local_abs, sin_clave)
        visit_without = self._sin_clave_rate(p.visitante_id, visit_abs, sin_clave)

        triple = h2h.get((p.local_id, p.visitante_id), [0, 0, 0])

        return [
            local_form["points"],
            local_form["goal_diff"],
            visit_form["points"],
            visit_form["goal_diff"],
            self._acumulado(h_local, "pts_pm"),
            self._acumulado(h_local, "gf_pm"),
            self._acumulado(h_local, "gc_pm"),
            self._acumulado(h_visitante, "pts_pm"),
            self._acumulado(h_visitante, "gf_pm"),
            self._acumulado(h_visitante, "gc_pm"),
            len(local_abs) * self.peso_jugador_clave,
            len(visit_abs) * self.peso_jugador_clave,
            local_without,
            visit_without,
            triple[0],
            triple[1],
            triple[2],
            self.home_bias,
        ]

    def _forma(self, historial_equipo):
        """Puntos y diferencia de goles en los últimos N partidos jugados."""
        recientes = historial_equipo[-self.form_window :]
        return {
            "points": sum(m["puntos"] for m in recientes),
            "goal_diff": sum((m["gf"] - m["ga"]) for m in recientes),
        }

    def _acumulado(self, historial_equipo, campo):
        """Promedios acumulados de la temporada hasta antes del partido."""
        n = len(historial_equipo)
        if n == 0:
            return 0.0
        if campo == "pts_pm":
            return sum(m["puntos"] for m in historial_equipo) / n
        if campo == "gf_pm":
            return sum(m["gf"] for m in historial_equipo) / n
        return sum(m["ga"] for m in historial_equipo) / n

    def _ausentes_importantes(self, p, equipo_id):
        return {
            a.jugador_id
            for a in p.ausencias.all()
            if a.jugador.es_importante and a.jugador.activo and a.jugador.equipo_id == equipo_id
        }

    def _sin_clave_rate(self, equipo_id, ausentes, sin_clave):
        """Rendimiento medio del equipo en partidos donde faltaron esos jugadores clave."""
        tasas = [
            sin_clave["por_jugador"][(equipo_id, jugador_id)]
            for jugador_id in ausentes
            if (equipo_id, jugador_id) in sin_clave["por_jugador"]
        ]
        if tasas:
            return sum(tasas) / len(tasas)
        return sin_clave["global"].get(equipo_id, 0.0)