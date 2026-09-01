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

    def generar_desde_quiniela(self, quiniela: Any) -> Dict[str, Any]:
        """Genera el boleto a partir del modelo Quiniela respetando el orden oficial de las 15 casillas."""
        casillas = list(quiniela.casillas.select_related("partido__local", "partido__visitante", "partido__prediccion").order_by("posicion"))
        if not casillas:
            return {
                "quiniela": quiniela,
                "filas": [],
                "n_dobles": self.n_dobles,
                "n_triples": self.n_triples,
                "total_columnas": self.total_columnas,
                "coste_total": self.coste_total,
                "pleno_al_15": None,
                "total_aciertos": 0,
                "partidos_jugados": 0,
            }

        # Dixon-Coles model loader
        dixon_coles = None
        try:
            from ligas.ml.predictor import Predictor
            predictor = Predictor.cargar()
            dixon_coles = predictor.dixon_coles
        except Exception:
            dixon_coles = None

        partidos_items = []
        for c in casillas:
            p = c.partido
            pred = getattr(p, "prediccion", None)
            prob_l = pred.prob_local if pred else 0.45
            prob_e = pred.prob_empate if pred else 0.28
            prob_v = pred.prob_visitante if pred else 0.27

            pleno_info = None
            if dixon_coles is not None:
                try:
                    pleno_info = dixon_coles.predict_pleno_al_15(p.local_id, p.visitante_id)
                except Exception:
                    pleno_info = None

            partidos_items.append({
                "casilla_id": c.id,
                "posicion": c.posicion,
                "partido": p,
                "local": p.local,
                "visitante": p.visitante,
                "prob_local": prob_l,
                "prob_empate": prob_e,
                "prob_visitante": prob_v,
                "pleno_info": pleno_info,
            })

        bloque_14 = [item for item in partidos_items if item["posicion"] <= 14]
        partido_15 = next((item for item in partidos_items if item["posicion"] == 15), None)

        filas = []
        aciertos_14 = 0
        partidos_jugados_14 = 0

        for item in bloque_14:
            p = item["partido"]
            p_l = item["prob_local"]
            p_e = item["prob_empate"]
            p_v = item["prob_visitante"]
            h = calcular_entropia(p_l, p_e, p_v)
            orden = obtener_signos_ordenados(p_l, p_e, p_v)

            fila_data = {
                "casilla_id": item["casilla_id"],
                "numero": item["posicion"],
                "partido": p,
                "local": item["local"],
                "visitante": item["visitante"],
                "division_nivel": p.division_nivel,
                "division_nombre": p.division_nombre,
                "prob_local": p_l,
                "prob_empate": p_e,
                "prob_visitante": p_v,
                "prob_local_pct": round(p_l * 100, 1),
                "prob_empate_pct": round(p_e * 100, 1),
                "prob_visitante_pct": round(p_v * 100, 1),
                "entropia": h,
                "signos_ordenados": orden,
                "signo_base": orden[0][0],
                "tipo_apuesta": "FIJO",
                "signos_jugados": [orden[0][0]],
                "jugado": p.jugado,
                "resultado_real": p.resultado,
                "goles_local": p.goles_local,
                "goles_visitante": p.goles_visitante,
                "acierto": None,
            }
            filas.append(fila_data)

        # Asignar dobles y triples por entropía
        indices_por_entropia = sorted(
            range(len(filas)),
            key=lambda i: filas[i]["entropia"],
            reverse=True,
        )

        for i, idx in enumerate(indices_por_entropia):
            if i < self.n_triples:
                filas[idx]["tipo_apuesta"] = "TRIPLE"
                filas[idx]["signos_jugados"] = ["1", "X", "2"]
            elif i < (self.n_triples + self.n_dobles):
                filas[idx]["tipo_apuesta"] = "DOBLE"
                orden = filas[idx]["signos_ordenados"]
                filas[idx]["signos_jugados"] = sorted([orden[0][0], orden[1][0]])
            else:
                filas[idx]["tipo_apuesta"] = "FIJO"
                filas[idx]["signos_jugados"] = [filas[idx]["signo_base"]]

            # Evaluar si el partido ya se ha jugado
            if filas[idx]["jugado"]:
                partidos_jugados_14 += 1
                es_acierto = filas[idx]["resultado_real"] in filas[idx]["signos_jugados"]
                filas[idx]["acierto"] = es_acierto
                if es_acierto:
                    aciertos_14 += 1

        # Pleno al 15
        pleno_data = None
        pleno_acierto = None
        if partido_15 is not None:
            p = partido_15["partido"]
            pleno_info = partido_15.get("pleno_info")
            p_l = partido_15["prob_local"]
            p_e = partido_15["prob_empate"]
            p_v = partido_15["prob_visitante"]
            pronostico_goles = pleno_info.get("pleno_recomendado", "1-0") if pleno_info else "1-0"
            marcador_real = f"{p.goles_local if p.goles_local is not None and p.goles_local < 3 else 'M'}-{p.goles_visitante if p.goles_visitante is not None and p.goles_visitante < 3 else 'M'}" if p.jugado else None
            if p.jugado:
                pleno_acierto = (marcador_real == pronostico_goles)

            pleno_data = {
                "casilla_id": partido_15["casilla_id"],
                "numero": 15,
                "partido": p,
                "local": partido_15["local"],
                "visitante": partido_15["visitante"],
                "division_nivel": p.division_nivel,
                "division_nombre": p.division_nombre,
                "prob_local_pct": round(p_l * 100, 1),
                "prob_empate_pct": round(p_e * 100, 1),
                "prob_visitante_pct": round(p_v * 100, 1),
                "pronostico_goles": pronostico_goles,
                "marcador_exacto": pleno_info.get("marcador_probable", (1, 0)) if pleno_info else (1, 0),
                "marcador_prob": round((pleno_info.get("marcador_probable_prob", 0.20) if pleno_info else 0.20) * 100, 1),
                "cat_local": pleno_info.get("cat_local") if pleno_info else {"0": 0.2, "1": 0.5, "2": 0.2, "M": 0.1},
                "cat_visit": pleno_info.get("cat_visit") if pleno_info else {"0": 0.4, "1": 0.4, "2": 0.15, "M": 0.05},
                "jugado": p.jugado,
                "marcador_real": marcador_real,
                "goles_local": p.goles_local,
                "goles_visitante": p.goles_visitante,
                "acierto": pleno_acierto,
            }

        total_aciertos = aciertos_14 + (1 if pleno_acierto else 0)
        total_jugados = partidos_jugados_14 + (1 if (pleno_data and pleno_data["jugado"]) else 0)

        return {
            "quiniela": quiniela,
            "filas": filas,
            "n_dobles": self.n_dobles,
            "n_triples": self.n_triples,
            "total_columnas": self.total_columnas,
            "coste_total": self.coste_total,
            "pleno_al_15": pleno_data,
            "aciertos_14": aciertos_14,
            "total_aciertos": total_aciertos,
            "partidos_jugados": total_jugados,
            "total_partidos": len(filas) + (1 if pleno_data else 0),
        }


def aplicar_predicciones_a_quiniela(quiniela: Any) -> None:
    """Ejecuta el pipeline ML sobre los partidos de la Quiniela y persiste los pronósticos en cada CasillaQuiniela."""
    from ligas.ml.predictor import predecir_partidos
    from ligas.models import CasillaQuiniela

    # Asegurar predicciones ML específicamente en los partidos grabados en las casillas de la quiniela
    casillas = list(
        quiniela.casillas.select_related("partido__local", "partido__visitante", "partido__prediccion")
        .order_by("posicion")
    )
    partidos = [c.partido for c in casillas if c.partido]

    if partidos:
        try:
            predecir_partidos(partidos, temporada=quiniela.temporada)
        except Exception:
            pass

    generador = GeneradorQuiniela(n_dobles=quiniela.n_dobles, n_triples=quiniela.n_triples)
    boleto = generador.generar_desde_quiniela(quiniela)

    # Persistir en las casillas los pronósticos calculados respetando los partidos asignados
    for fila in boleto["filas"]:
        CasillaQuiniela.objects.filter(pk=fila["casilla_id"]).update(
            signo_base=fila["signo_base"],
            tipo_apuesta=fila["tipo_apuesta"],
            signos_jugados="".join(fila["signos_jugados"]),
        )

    if boleto["pleno_al_15"]:
        p15 = boleto["pleno_al_15"]
        CasillaQuiniela.objects.filter(pk=p15["casilla_id"]).update(
            signo_base="1",
            tipo_apuesta="FIJO",
            signos_jugados="1",
            pronostico_pleno=p15["pronostico_goles"],
        )

    quiniela.evaluar_aciertos()


def poblar_casillas_oficiales(quiniela: Any) -> int:
    """Auto-asigna los 15 partidos oficiales de la jornada (1ª Masc, 2ª Masc y Liga F coincidentes)."""
    from datetime import timedelta
    from ligas.models import CasillaQuiniela, Jornada, Partido

    temporada = quiniela.temporada
    jornada = quiniela.jornada

    # Si no tiene jornada asociada explícitamente, intentar buscarla por número en su temporada
    if not jornada and temporada:
        jornada = Jornada.objects.filter(temporada=temporada, numero=quiniela.numero).first()
        if jornada:
            quiniela.jornada = jornada
            quiniela.save(update_fields=["jornada"])

    if not jornada and not temporada:
        return 0

    qs = Partido.objects.select_related("local", "visitante", "jornada__temporada").prefetch_related(
        "local__participaciones__division", "visitante__participaciones__division"
    )

    partidos_dict = {}

    if jornada is not None:
        directos = list(qs.filter(jornada=jornada).order_by("fecha", "id"))
        for p in directos:
            partidos_dict[p.id] = p

        fechas = [p.fecha for p in directos if p.fecha]
        if fechas:
            f_min = min(fechas) - timedelta(days=2)
            f_max = max(fechas) + timedelta(days=2)
            coincidentes = list(
                qs.filter(
                    jornada__temporada=jornada.temporada,
                    fecha__gte=f_min,
                    fecha__lte=f_max,
                ).order_by("fecha", "id")
            )
            for p in coincidentes:
                partidos_dict.setdefault(p.id, p)
        else:
            p_fem_extra = list(
                qs.filter(
                    jornada__temporada=jornada.temporada,
                    local__participaciones__division__categoria="FEM",
                ).order_by("jornada__numero", "fecha", "id")[:6]
            )
            for p in p_fem_extra:
                partidos_dict.setdefault(p.id, p)
    else:
        partidos_temp = list(qs.filter(jornada__temporada=temporada).order_by("fecha", "id"))
        for p in partidos_temp:
            partidos_dict[p.id] = p

    partidos_qs = list(partidos_dict.values())

    p_div1_masc = [
        p for p in partidos_qs if p.division and getattr(p.division, "categoria", "MASC") == "MASC" and p.division.nivel == 1
    ]
    p_div2_masc = [
        p for p in partidos_qs if p.division and getattr(p.division, "categoria", "MASC") == "MASC" and p.division.nivel == 2
    ]
    p_fem = [
        p for p in partidos_qs if p.division and getattr(p.division, "categoria", "MASC") == "FEM"
    ]
    p_otros = [p for p in partidos_qs if p not in (p_div1_masc + p_div2_masc + p_fem)]

    # Priorización oficial 1X2: 10 de 1ª División, 3-4 de 2ª División y 1-2 de Liga Femenina
    elegidos = p_div1_masc[:10]
    if p_div2_masc:
        elegidos += p_div2_masc[: (5 if not p_fem else 3)]
    if p_fem:
        espacios_fem = max(1, 15 - len(elegidos))
        elegidos += p_fem[:espacios_fem]

    if len(elegidos) < 15:
        restantes = [p for p in (p_div1_masc[10:] + p_div2_masc + p_fem + p_otros) if p not in elegidos]
        elegidos += restantes[: (15 - len(elegidos))]

    # Guardar en base de datos
    for idx, partido in enumerate(elegidos[:15], start=1):
        CasillaQuiniela.objects.update_or_create(
            quiniela=quiniela,
            posicion=idx,
            defaults={"partido": partido},
        )

    if elegidos:
        aplicar_predicciones_a_quiniela(quiniela)

    return len(elegidos[:15])
