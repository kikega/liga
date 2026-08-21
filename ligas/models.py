from datetime import date

from django.core.exceptions import ValidationError
from django.db import models

PUNTOS_VICTORIA = 3
PUNTOS_EMPATE = 1
PUNTOS_DERROTA = 0

RESULTADO_LOCAL = "1"
RESULTADO_EMPATE = "X"
RESULTADO_VISITANTE = "2"

RESULTADO_CHOICES = [
    (RESULTADO_LOCAL, "Gana Local"),
    (RESULTADO_EMPATE, "Empate"),
    (RESULTADO_VISITANTE, "Gana Visitante"),
]

MOTIVO_AUSENCIA_CHOICES = [
    ("lesion", "Lesión"),
    ("sancion", "Sanción"),
]


class Division(models.Model):
    """Competición: 1ª y 2ª División del fútbol español."""

    nombre = models.CharField(max_length=64, unique=True)
    nivel = models.PositiveSmallIntegerField(unique=True, help_text="1 para Primera, 2 para Segunda")
    n_descensos = models.PositiveSmallIntegerField(default=3, help_text="Equipos que descienden al final de temporada")

    class Meta:
        ordering = ["nivel"]

    def __str__(self):
        return self.nombre


class Temporada(models.Model):
    """Temporada, p. ej. '2024-2025'. Marca la frontera para ascensos/descensos."""

    nombre = models.CharField(max_length=32, unique=True)
    inicio = models.DateField()
    fin = models.DateField()
    activa = models.BooleanField(default=True)

    class Meta:
        ordering = ["-inicio"]

    def __str__(self):
        return self.nombre

    def partidos_historicos(self):
        """Partidos ya jugados de la temporada (para histórico de features/entrenamiento)."""
        return Partido.objects.filter(jornada__temporada=self, goles_local__isnull=False)

    def aplicar_ascensos_descensos(self):
        """Regla de negocio: 3 descienden de 1ª y 3 ascienden de 2ª al cierre.

        Crea/actualiza las Participaciones de la temporada siguiente según la
        clasificación final de la temporada actual.
        """
        from ligas.services import clasificacion_por_division

        temporada_siguiente = (
            Temporada.objects.filter(inicio__gt=self.inicio).order_by("inicio").first()
        )
        if temporada_siguiente is None:
            raise ValidationError(
                "No existe una temporada posterior a la que aplicar los ascensos/descensos."
            )

        movimientos = []
        for division in Division.objects.order_by("nivel"):
            clasificacion = clasificacion_por_division(self, division)
            if not clasificacion:
                continue
            equipos = [fila["equipo_id"] for fila in clasificacion]
            n_descensos = division.n_descensos
            descendidos = equipos[-n_descensos:] if len(equipos) >= n_descensos else []
            for pos, equipo in enumerate(equipos, start=1):
                Participacion.objects.filter(temporada=self, equipo_id=equipo).update(
                    posicion_final=pos
                )

            division_inferior = self._division_siguiente(division, subir=False)
            if division_inferior is not None:
                # Los que no descienden se mantienen en la misma división.
                for equipo in equipos[: len(equipos) - len(descendidos)]:
                    Participacion.objects.update_or_create(
                        temporada=temporada_siguiente,
                        equipo_id=equipo,
                        defaults={"division": division},
                    )
                for equipo in descendidos:
                    movimientos.append((equipo, division, "descenso"))
                    Participacion.objects.update_or_create(
                        temporada=temporada_siguiente,
                        equipo_id=equipo,
                        defaults={"division": division_inferior},
                    )
            else:
                # No hay división inferior: todos se mantienen.
                for equipo in equipos:
                    Participacion.objects.update_or_create(
                        temporada=temporada_siguiente,
                        equipo_id=equipo,
                        defaults={"division": division},
                    )

        division_1 = Division.objects.filter(nivel=1).first()
        division_2 = Division.objects.filter(nivel=2).first()
        if division_2 and division_1:
            ascendidos = [
                fila["equipo_id"]
                for fila in clasificacion_por_division(self, division_2)[: division_1.n_descensos]
            ]
            for equipo in ascendidos:
                movimientos.append((equipo, division_2, "ascenso"))
                Participacion.objects.update_or_create(
                    temporada=temporada_siguiente,
                    equipo_id=equipo,
                    defaults={"division": division_1},
                )
        return movimientos

    @classmethod
    def iniciar_siguiente(cls):
        """Crea la temporada siguiente a la más reciente y le da de alta los equipos.

        - Nombra la nueva temporada a partir del año de la última ('2026-2027').
        - Marca como activa solo la nueva temporada.
        - Aplica ascensos/descensos desde la temporada anterior y crea la Jornada 1.
        """
        ultima = cls.objects.order_by("-inicio").first()
        anio = (ultima.inicio.year + 1) if ultima is not None else date.today().year
        nombre = f"{anio}-{anio + 1}"
        temporada, creada = cls.objects.get_or_create(
            nombre=nombre,
            defaults={"inicio": date(anio, 8, 1), "fin": date(anio + 1, 7, 31), "activa": True},
        )
        cls.objects.exclude(pk=temporada.pk).update(activa=False)
        temporada.activa = True
        temporada.save(update_fields=["activa"])

        movimientos = []
        if ultima is not None and ultima.pk != temporada.pk:
            try:
                movimientos = ultima.aplicar_ascensos_descensos()
            except ValidationError:
                movimientos = []
        Jornada.objects.get_or_create(temporada=temporada, numero=1)
        return temporada, creada, movimientos

    @staticmethod
    def _division_siguiente(division, subir=False):
        nivel = division.nivel + 1
        return Division.objects.filter(nivel=nivel).first()


class Equipo(models.Model):
    nombre = models.CharField(max_length=64, unique=True)
    escudo = models.ImageField(
        upload_to="escudos/",
        null=True,
        blank=True,
        help_text="Imagen o icono del escudo del equipo.",
    )
    color_primario = models.CharField(
        max_length=7,
        default="#1e293b",
        blank=True,
        help_text="Color principal en formato #RRGGBB (acento en la UI).",
    )
    color_secundario = models.CharField(
        max_length=7,
        default="#0f172a",
        blank=True,
        help_text="Color secundario en formato #RRGGBB.",
    )

    class Meta:
        ordering = ["nombre"]

    def __str__(self):
        return self.nombre


class Participacion(models.Model):
    """Equipo en una temporada concreta en una división.

    Permite modelar que un equipo descienda/ascienda entre temporadas.
    """

    temporada = models.ForeignKey(Temporada, on_delete=models.CASCADE, related_name="participaciones")
    equipo = models.ForeignKey(Equipo, on_delete=models.CASCADE, related_name="participaciones")
    division = models.ForeignKey(Division, on_delete=models.CASCADE, related_name="participaciones")
    posicion_final = models.PositiveSmallIntegerField(null=True, blank=True)

    class Meta:
        unique_together = [("temporada", "equipo")]
        ordering = ["temporada", "division", "posicion_final"]

    def __str__(self):
        return f"{self.equipo} · {self.temporada} · {self.division}"


class Jugador(models.Model):
    """Jugador con flag de jugador clave (es_importante)."""

    nombre = models.CharField(max_length=100)
    equipo = models.ForeignKey(Equipo, on_delete=models.CASCADE, related_name="jugadores")
    es_importante = models.BooleanField(default=False, help_text="Jugador clave para su equipo")
    activo = models.BooleanField(default=True)

    class Meta:
        ordering = ["equipo", "nombre"]

    def __str__(self):
        return self.nombre


class Jornada(models.Model):
    temporada = models.ForeignKey(Temporada, on_delete=models.CASCADE, related_name="jornadas")
    numero = models.PositiveSmallIntegerField()
    cerrada = models.BooleanField(default=False, help_text="Resultados registrados y modelo re-entrenado")

    class Meta:
        unique_together = [("temporada", "numero")]
        ordering = ["temporada", "numero"]

    def __str__(self):
        return f"Jornada {self.numero} · {self.temporada}"


class Partido(models.Model):
    jornada = models.ForeignKey(Jornada, on_delete=models.CASCADE, related_name="partidos")
    local = models.ForeignKey(Equipo, on_delete=models.CASCADE, related_name="partidos_como_local")
    visitante = models.ForeignKey(Equipo, on_delete=models.CASCADE, related_name="partidos_como_visitante")
    goles_local = models.PositiveSmallIntegerField(null=True, blank=True)
    goles_visitante = models.PositiveSmallIntegerField(null=True, blank=True)
    fecha = models.DateField(null=True, blank=True)

    class Meta:
        unique_together = [("jornada", "local", "visitante")]
        ordering = ["jornada", "fecha", "id"]

    @property
    def fecha_formateada(self) -> str:
        """Devuelve la fecha en español con formato 'Viernes, 16 de agosto de 2024'."""
        if not self.fecha:
            return "Fecha por confirmar"
        dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        meses = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
        ]
        d = self.fecha
        dia_sem = dias[d.weekday()]
        mes = meses[d.month - 1]
        return f"{dia_sem}, {d.day} de {mes} de {d.year}"

    @property
    def jugado(self):
        return self.goles_local is not None and self.goles_visitante is not None

    @property
    def resultado(self):
        """Devuelve '1', 'X', '2' o None si no se ha jugado."""
        if not self.jugado:
            return None
        if self.goles_local > self.goles_visitante:
            return RESULTADO_LOCAL
        if self.goles_local == self.goles_visitante:
            return RESULTADO_EMPATE
        return RESULTADO_VISITANTE

    def puntos_local(self):
        return self._puntos(self.goles_local, self.goles_visitante)

    def puntos_visitante(self):
        return self._puntos(self.goles_visitante, self.goles_local)

    @staticmethod
    def _puntos(goles_propios, goles_rival):
        """Regla estricta: 3 al ganador, 1 por empate, 0 al perdedor."""
        if goles_propios is None or goles_rival is None:
            return None
        if goles_propios > goles_rival:
            return PUNTOS_VICTORIA
        if goles_propios == goles_rival:
            return PUNTOS_EMPATE
        return PUNTOS_DERROTA

    def guardar_resultado(self, goles_local, goles_visitante):
        self.goles_local = goles_local
        self.goles_visitante = goles_visitante
        self.save(update_fields=["goles_local", "goles_visitante"])

    @property
    def division(self):
        """Devuelve el objeto Division del partido según la participación del local en su temporada."""
        temporada_id = self.jornada.temporada_id
        for participacion in self.local.participaciones.all():
            if participacion.temporada_id == temporada_id:
                return participacion.division
        ultima = self.local.participaciones.order_by("-temporada__inicio").first()
        return ultima.division if ultima else None

    @property
    def division_nivel(self) -> int:
        div = self.division
        return div.nivel if div else 1

    @property
    def division_nombre(self) -> str:
        div = self.division
        return div.nombre if div else "1ª División"

    @property
    def division_temporada_id(self):
        """División del partido según la participación del local en su temporada."""
        div = self.division
        return div.id if div else None


class Ausencia(models.Model):
    """Baja de un jugador en un partido (lesión o sanción)."""

    partido = models.ForeignKey(Partido, on_delete=models.CASCADE, related_name="ausencias")
    jugador = models.ForeignKey(Jugador, on_delete=models.CASCADE, related_name="ausencias")
    motivo = models.CharField(max_length=16, choices=MOTIVO_AUSENCIA_CHOICES)

    class Meta:
        unique_together = [("partido", "jugador")]

    def __str__(self):
        return f"{self.jugador} baja ({self.motivo}) en {self.partido}"


class Prediccion(models.Model):
    """Predicción del modelo ML sobre un partido (1/X/2 + probabilidades)."""

    partido = models.OneToOneField(Partido, on_delete=models.CASCADE, related_name="prediccion")
    prob_local = models.FloatField()
    prob_empate = models.FloatField()
    prob_visitante = models.FloatField()
    resultado_predicho = models.CharField(max_length=1, choices=RESULTADO_CHOICES)
    resultado_real = models.CharField(max_length=1, choices=RESULTADO_CHOICES, null=True, blank=True)
    acierto = models.BooleanField(null=True, blank=True)
    fecha_prediccion = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-fecha_prediccion"]

    def __str__(self):
        return f"{self.partido} -> {self.resultado_predicho} ({self.prob_local:.0%}/{self.prob_empate:.0%}/{self.prob_visitante:.0%})"

    @property
    def probabilidades(self):
        return (self.prob_local, self.prob_empate, self.prob_visitante)

    @property
    def prob_local_pct(self) -> float:
        return round(self.prob_local * 100, 1)

    @property
    def prob_empate_pct(self) -> float:
        return round(self.prob_empate * 100, 1)

    @property
    def prob_visitante_pct(self) -> float:
        return round(self.prob_visitante * 100, 1)

    @property
    def max_prob(self) -> float:
        return max(self.prob_local, self.prob_empate, self.prob_visitante)

    @property
    def confianza_nivel(self) -> str:
        """Nivel de certeza estadística del pronóstico."""
        p = self.max_prob
        if p >= 0.55:
            return "ALTA"
        if p >= 0.42:
            return "MEDIA"
        return "DISPUTADO"

    def actualizar_con_resultado(self, resultado):
        """Registra el resultado real y si la predicción fue correcta."""
        self.resultado_real = resultado
        self.acierto = resultado == self.resultado_predicho
        self.save(update_fields=["resultado_real", "acierto"])
        return self.acierto


class Configuracion(models.Model):
    """Configuración global de la aplicación (patrón singleton, id=1)."""

    temporada_actual = models.ForeignKey(
        Temporada,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
        help_text="Temporada que se muestra por defecto en el dashboard.",
    )
    proxima_jornada = models.ForeignKey(
        Jornada,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
        help_text="Próxima jornada para predecir resultados.",
    )

    class Meta:
        verbose_name = "Configuración"
        verbose_name_plural = "Configuración"

    def __str__(self):
        return "Configuración general"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def cargar(cls):
        """Devuelve la configuración (la crea si no existe)."""
        config, _ = cls.objects.get_or_create(pk=1)
        return config
