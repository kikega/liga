"""Motor de análisis y generación inteligente de boletos para la Quiniela Española (1X2 + Pleno al 15).

Calcula la entropía de incertidumbre de cada pronóstico para sugerir de forma óptima
apuestas simples, dobles y triples optimizando el coste y la probabilidad de éxito.
"""

import math
from typing import Any, Dict, List, Tuple
import numpy as np

PRECIO_COLUMNA = 0.75  # Precio oficial por apuesta simple en la Quiniela española (€)


def calcular_entropia(p_local: float, p_empate: float, p_visitante: float) -> float:
    """Calcula la entropía de Shannon H(p) en bits.

    Valores cercanos a 0 indican máxima certeza (ideal para signo fijo).
    Valores cercanos a 1.585 indican máxima incertidumbre (ideal para triple/doble).
    """
    probs = [p_local, p_empate, p_visitante]
    entropia = 0.0
    for p in probs:
        if p > 1e-6:
            entropia -= p * math.log2(p)
    return float(entropia)


def obtener_signos_ordenados(p_local: float, p_empate: float, p_visitante: float) -> List[Tuple[str, float]]:
    """Devuelve la lista de signos [('1', prob), ('X', prob), ('2', prob)] ordenada por mayor probabilidad."""
    lista = [("1", p_local), ("X", p_empate), ("2", p_visitante)]
    return sorted(lista, key=lambda item: item[1], reverse=True)


class GeneradorQuiniela:
    """Generador y optimizador de apuestas de Quiniela."""

    def __init__(self, n_dobles: int = 2, n_triples: int = 1) -> None:
        self.n_dobles = max(0, min(n_dobles, 14))
        self.n_triples = max(0, min(n_triples, 14 - self.n_dobles))

    @property
    def total_columnas(self) -> int:
        """Número total de apuestas simples generadas."""
        return (2**self.n_dobles) * (3**self.n_triples)

    @property
    def coste_total(self) -> float:
        """Coste en euros de la combinación jugada."""
        return round(self.total_columnas * PRECIO_COLUMNA, 2)

    def generar_boleto(self, partidos_con_prediccion: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera el boleto de Quiniela con 14 partidos + Pleno al 15.

        Ordena los partidos por entropía para asignar los triples a los partidos más
        inciertos, los dobles a los siguientes más inciertos, y fijos al resto.
        """
        if not partidos_con_prediccion:
            return {
                "filas": [],
                "n_dobles": self.n_dobles,
                "n_triples": self.n_triples,
                "total_columnas": self.total_columnas,
                "coste_total": self.coste_total,
                "pleno_al_15": None,
            }

        # 14 primeros partidos corresponden al bloque 1X2; el 15 es el Pleno al 15
        bloque_14 = partidos_con_prediccion[:14]
        partido_15 = partidos_con_prediccion[14] if len(partidos_con_prediccion) >= 15 else (partidos_con_prediccion[-1] if len(partidos_con_prediccion) == 15 else None)

        # Enriquecer cada partido con métricas de incertidumbre
        items_analizados = []
        for idx, item in enumerate(bloque_14, start=1):
            p_l = item["prob_local"]
            p_e = item["prob_empate"]
            p_v = item["prob_visitante"]
            h = calcular_entropia(p_l, p_e, p_v)
            orden = obtener_signos_ordenados(p_l, p_e, p_v)

            items_analizados.append(
                {
                    "numero": idx,
                    "partido": item.get("partido"),
                    "local": item.get("local"),
                    "visitante": item.get("visitante"),
                    "prob_local": p_l,
                    "prob_empate": p_e,
                    "prob_visitante": p_v,
                    "prob_local_pct": round(p_l * 100, 1),
                    "prob_empate_pct": round(p_e * 100, 1),
                    "prob_visitante_pct": round(p_v * 100, 1),
                    "entropia": h,
                    "signos_ordenados": orden,
                    "signo_base": orden[0][0],
                    "prob_base": orden[0][1],
                    "tipo_apuesta": "FIJO",
                    "signos_jugados": [orden[0][0]],
                }
            )

        # Asignación óptima de dobles y triples según entropía decreciente
        indices_por_entropia = sorted(
            range(len(items_analizados)),
            key=lambda i: items_analizados[i]["entropia"],
            reverse=True,
        )

        for i, idx in enumerate(indices_por_entropia):
            if i < self.n_triples:
                items_analizados[idx]["tipo_apuesta"] = "TRIPLE"
                items_analizados[idx]["signos_jugados"] = ["1", "X", "2"]
            elif i < (self.n_triples + self.n_dobles):
                items_analizados[idx]["tipo_apuesta"] = "DOBLE"
                orden = items_analizados[idx]["signos_ordenados"]
                items_analizados[idx]["signos_jugados"] = sorted([orden[0][0], orden[1][0]])
            else:
                items_analizados[idx]["tipo_apuesta"] = "FIJO"
                items_analizados[idx]["signos_jugados"] = [items_analizados[idx]["signo_base"]]

        # Análisis específico para el Pleno al 15
        pleno_data = None
        if partido_15 is not None:
            pleno_info = partido_15.get("pleno_info")
            p_l = partido_15["prob_local"]
            p_e = partido_15["prob_empate"]
            p_v = partido_15["prob_visitante"]
            pleno_data = {
                "numero": 15,
                "partido": partido_15.get("partido"),
                "local": partido_15.get("local"),
                "visitante": partido_15.get("visitante"),
                "prob_local_pct": round(p_l * 100, 1),
                "prob_empate_pct": round(p_e * 100, 1),
                "prob_visitante_pct": round(p_v * 100, 1),
                "pronostico_goles": pleno_info.get("pleno_recomendado", "1-0") if pleno_info else "1-0",
                "marcador_exacto": pleno_info.get("marcador_probable", (1, 0)) if pleno_info else (1, 0),
                "marcador_prob": round((pleno_info.get("marcador_probable_prob", 0.20) if pleno_info else 0.20) * 100, 1),
                "cat_local": pleno_info.get("cat_local") if pleno_info else {"0": 0.2, "1": 0.5, "2": 0.2, "M": 0.1},
                "cat_visit": pleno_info.get("cat_visit") if pleno_info else {"0": 0.4, "1": 0.4, "2": 0.15, "M": 0.05},
            }

        return {
            "filas": items_analizados,
            "n_dobles": self.n_dobles,
            "n_triples": self.n_triples,
            "total_columnas": self.total_columnas,
            "coste_total": self.coste_total,
            "pleno_al_15": pleno_data,
        }
