"""Cierra una jornada: registra resultados, evalúa predicciones y re-entrena.

Uso:

    python manage.py close_jornada --temporada 2024-2025 --jornada 23

Pasos:
1. Actualiza cada ``Prediccion`` de la jornada con el resultado real y su acierto.
2. Marca la jornada como cerrada.
3. Re-entrena el modelo con todo el histórico (re-entrenamiento continuo).
"""

from django.core.management.base import BaseCommand, CommandError

from ligas.ml.predictor import entrenar_modelo
from ligas.models import Jornada, Partido, Temporada


class Command(BaseCommand):
    help = "Cierra la jornada, evalúa predicciones y re-entrena el modelo."

    def add_arguments(self, parser):
        parser.add_argument("--temporada", required=True, help="Nombre de la temporada.")
        parser.add_argument("--jornada", required=True, type=int, help="Número de jornada.")
        parser.add_argument("--output", help="Ruta del archivo joblib de salida.")

    def handle(self, *args, **options):
        try:
            temporada = Temporada.objects.get(nombre=options["temporada"])
        except Temporada.DoesNotExist as exc:
            raise CommandError(f"No existe la temporada {options['temporada']!r}.") from exc
        try:
            jornada = Jornada.objects.get(temporada=temporada, numero=options["jornada"])
        except Jornada.DoesNotExist as exc:
            raise CommandError(f"No existe la jornada {options['jornada']} de {temporada}.") from exc

        aciertos = total = 0
        for partido in jornada.partidos.select_related("prediccion").filter(goles_local__isnull=False):
            if not hasattr(partido, "prediccion"):
                continue
            total += 1
            if partido.prediccion.actualizar_con_resultado(partido.resultado):
                aciertos += 1

        jornada.cerrada = True
        jornada.save(update_fields=["cerrada"])

        self.stdout.write(
            self.style.SUCCESS(
                f"Jornada {jornada.numero} · {temporada}: {aciertos}/{total} predicciones acertadas."
            )
        )

        partidos = list(
            Partido.objects.filter(goles_local__isnull=False)
            .select_related("jornada__temporada", "local", "visitante")
            .prefetch_related("ausencias__jugador")
        )
        try:
            predictor, exactitud, _, ruta = entrenar_modelo(partidos, output=options["output"])
        except ValueError as exc:
            self.stderr.write(self.style.WARNING(f"No se re-entrenó el modelo: {exc}"))
            return

        self.stdout.write(
            self.style.SUCCESS(f"Modelo re-entrenado en {ruta} (exactitud {exactitud:.1%}).")
        )