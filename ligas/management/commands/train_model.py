"""Entrena y persiste el modelo predictivo con el histórico de partidos.

Uso:

    python manage.py train_model [--temporada 2024-2025] [--ventana 5] [--output ruta.joblib]

Sin ``--temporada`` entrena con todo el histórico disponible en la base de datos.
"""

from django.core.management.base import BaseCommand, CommandError

from ligas.ml.predictor import entrenar_modelo
from ligas.models import Partido


class Command(BaseCommand):
    help = "Entrena el modelo RandomForest (1/X/2) con el histórico y lo persiste con joblib."

    def add_arguments(self, parser):
        parser.add_argument("--temporada", help="Nombre de la temporada (por defecto: todo el histórico).")
        parser.add_argument("--ventana", type=int, default=5, help="Partidos de forma reciente considerados.")
        parser.add_argument("--output", help="Ruta del archivo joblib de salida.")
        parser.add_argument("--min-muestras", type=int, default=30, help="Mínimo de partidos jugados para entrenar.")

    def handle(self, *args, **options):
        consulta = (
            Partido.objects.filter(goles_local__isnull=False)
            .select_related("jornada__temporada", "local", "visitante")
            .prefetch_related("ausencias__jugador")
        )
        if options["temporada"]:
            consulta = consulta.filter(jornada__temporada__nombre=options["temporada"])

        partidos = list(consulta)
        if not partidos:
            raise CommandError("No hay partidos jugados en la base de datos para entrenar.")

        self.stdout.write(f"Características sobre {len(partidos)} partidos jugados...")
        try:
            predictor, exactitud, reporte, ruta = entrenar_modelo(
                partidos,
                form_window=options["ventana"],
                output=options["output"],
                min_muestras=options["min_muestras"],
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS(f"Modelo guardado en {ruta}"))
        self.stdout.write(
            self.style.SUCCESS(
                f"Exactitud en validación: {exactitud:.1%} "
                f"(muestras de validación: {reporte['macro avg']['support']})"
            )
        )