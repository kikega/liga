from datetime import date, datetime
from django import forms
from django.db.models import Max
from django.utils.dateparse import parse_date, parse_datetime

from ligas.models import (
    CasillaQuiniela,
    Configuracion,
    Division,
    Equipo,
    Jornada,
    Jugador,
    Participacion,
    Partido,
    Quiniela,
    Temporada,
)

FORM_INPUT = (
    "w-full border border-slate-300 rounded-xl px-3 py-2 bg-white text-xs sm:text-sm text-slate-800 "
    "shadow-sm focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 focus:outline-none transition"
)
FORM_SELECT = FORM_INPUT
FORM_CHECK = "h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"

DATE_INPUT_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
]


def parse_fecha_flexible(fecha_val):
    """Parsea fechas de forma pura y devuelve un date de Python sin desfases horarios."""
    if not fecha_val:
        return None

    if isinstance(fecha_val, datetime):
        return fecha_val.date()

    if isinstance(fecha_val, date):
        return fecha_val

    fecha_str = str(fecha_val).strip()
    if not fecha_str:
        return None

    d = parse_date(fecha_str)
    if d is not None:
        return d

    dt = parse_datetime(fecha_str)
    if dt is not None:
        return dt.date()

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(fecha_str, fmt).date()
        except ValueError:
            continue
    return None


def obtener_equipos_choices(temporada=None):
    """Devuelve las opciones de equipos agrupadas limpiamente por división."""
    div1 = Division.objects.filter(nivel=1).first()
    div2 = Division.objects.filter(nivel=2).first()

    equipos_1 = []
    equipos_2 = []
    ids_usados = set()

    if temporada is not None:
        p1 = Participacion.objects.filter(temporada=temporada, division=div1).select_related("equipo").order_by("equipo__nombre")
        p2 = Participacion.objects.filter(temporada=temporada, division=div2).select_related("equipo").order_by("equipo__nombre")
        equipos_1 = [p.equipo for p in p1]
        equipos_2 = [p.equipo for p in p2]
        ids_usados = {e.id for e in equipos_1} | {e.id for e in equipos_2}

    if not equipos_1 and not equipos_2:
        equipos_1 = list(Equipo.objects.filter(participaciones__division__nivel=1).distinct().order_by("nombre"))
        equipos_2 = list(Equipo.objects.filter(participaciones__division__nivel=2).distinct().order_by("nombre"))
        ids_usados = {e.id for e in equipos_1} | {e.id for e in equipos_2}

    choices = [("", "Selecciona equipo")]
    if equipos_1:
        choices.append(("🏆 1ª División (LaLiga EA Sports)", [(e.id, e.nombre) for e in equipos_1]))
    if equipos_2:
        choices.append(("🥈 2ª División (LaLiga Hypermotion)", [(e.id, e.nombre) for e in equipos_2]))

    otros = list(Equipo.objects.exclude(id__in=ids_usados).order_by("nombre"))
    if otros:
        choices.append(("🛡️ Otros Equipos Registrados", [(e.id, e.nombre) for e in otros]))

    return choices


class ConfiguracionForm(forms.ModelForm):
    class Meta:
        model = Configuracion
        fields = ["temporada_actual", "proxima_jornada"]
        widgets = {
            "temporada_actual": forms.Select(attrs={"class": FORM_SELECT}),
            "proxima_jornada": forms.Select(attrs={"class": FORM_SELECT}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["temporada_actual"].queryset = Temporada.objects.all().order_by("-inicio")
        self.fields["temporada_actual"].label = "Temporada Activa"
        
        # Filtrar próximas jornadas según la temporada seleccionada o actual
        temporada = getattr(self.instance, "temporada_actual", None) or Temporada.objects.filter(activa=True).first()
        if temporada:
            self.fields["proxima_jornada"].queryset = Jornada.objects.filter(temporada=temporada).order_by("numero")
        else:
            self.fields["proxima_jornada"].queryset = Jornada.objects.none()
        self.fields["proxima_jornada"].label = "Próxima Jornada a Predecir (Quiniela & Dashboard)"
        self.fields["proxima_jornada"].empty_label = "Selecciona la jornada activa"


class JornadaForm(forms.ModelForm):
    numero = forms.IntegerField(
        required=False,
        min_value=1,
        widget=forms.NumberInput(attrs={"class": FORM_INPUT, "placeholder": "Auto (siguiente número)"}),
        label="Número de Jornada",
    )

    class Meta:
        model = Jornada
        fields = ["temporada", "numero"]
        widgets = {
            "temporada": forms.Select(attrs={"class": FORM_SELECT}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["temporada"].queryset = Temporada.objects.all().order_by("-inicio")
        self.fields["temporada"].label = "Temporada"

    def clean_numero(self):
        numero = self.cleaned_data.get("numero")
        temporada = self.cleaned_data.get("temporada")
        if numero is None and temporada is not None:
            ultimo = temporada.jornadas.aggregate(maximo=Max("numero"))["maximo"] or 0
            numero = ultimo + 1
        return numero


class EquipoForm(forms.ModelForm):
    division = forms.ModelChoiceField(
        queryset=Division.objects.all().order_by("nivel"),
        empty_label="— Sin división asignada (No participa) —",
        widget=forms.Select(attrs={"class": FORM_SELECT}),
        label="División en la Temporada Actual",
        required=False,
    )

    class Meta:
        model = Equipo
        fields = ["nombre", "division", "escudo", "color_primario", "color_secundario"]
        widgets = {
            "nombre": forms.TextInput(attrs={"class": FORM_INPUT}),
            "escudo": forms.ClearableFileInput(attrs={"class": FORM_INPUT}),
            "color_primario": forms.TextInput(attrs={"class": FORM_INPUT, "type": "color"}),
            "color_secundario": forms.TextInput(attrs={"class": FORM_INPUT, "type": "color"}),
        }

    def __init__(self, *args, temporada=None, **kwargs):
        self.temporada = temporada or Temporada.objects.filter(activa=True).first()
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk and self.temporada:
            p = Participacion.objects.filter(temporada=self.temporada, equipo=self.instance).first()
            if p:
                self.fields["division"].initial = p.division_id
            else:
                self.fields["division"].initial = None

    def save(self, commit=True):
        equipo = super().save(commit=commit)
        div = self.cleaned_data.get("division")
        if self.temporada:
            if div:
                Participacion.objects.update_or_create(
                    temporada=self.temporada,
                    equipo=equipo,
                    defaults={"division": div},
                )
            else:
                Participacion.objects.filter(
                    temporada=self.temporada,
                    equipo=equipo,
                ).delete()
        return equipo


class JugadorForm(forms.ModelForm):
    class Meta:
        model = Jugador
        fields = ["nombre", "es_importante", "activo"]
        widgets = {
            "nombre": forms.TextInput(attrs={"class": FORM_INPUT}),
            "es_importante": forms.CheckboxInput(attrs={"class": "h-4 w-4"}),
            "activo": forms.CheckboxInput(attrs={"class": "h-4 w-4"}),
        }


class PartidoForm(forms.ModelForm):
    local = forms.ModelChoiceField(
        queryset=Equipo.objects.all(),
        empty_label="Selecciona local",
        widget=forms.Select(attrs={"class": FORM_SELECT}),
    )
    visitante = forms.ModelChoiceField(
        queryset=Equipo.objects.all(),
        empty_label="Selecciona visitante",
        widget=forms.Select(attrs={"class": FORM_SELECT}),
    )
    fecha = forms.DateField(
        required=False,
        input_formats=DATE_INPUT_FORMATS,
        widget=forms.DateInput(
            format="%Y-%m-%d",
            attrs={"class": FORM_INPUT, "type": "date"},
        ),
    )

    class Meta:
        model = Partido
        fields = ["local", "visitante", "fecha", "goles_local", "goles_visitante"]
        widgets = {
            "goles_local": forms.NumberInput(attrs={"class": FORM_INPUT}),
            "goles_visitante": forms.NumberInput(attrs={"class": FORM_INPUT}),
        }

    def __init__(self, *args, temporada=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["local"].choices = obtener_equipos_choices(temporada)
        self.fields["visitante"].choices = obtener_equipos_choices(temporada)

    def clean_fecha(self):
        return parse_fecha_flexible(self.cleaned_data.get("fecha"))


class PartidoRowForm(forms.Form):
    local = forms.ModelChoiceField(
        queryset=Equipo.objects.all(),
        empty_label="Selecciona local",
        widget=forms.Select(attrs={"class": FORM_SELECT}),
    )
    visitante = forms.ModelChoiceField(
        queryset=Equipo.objects.all(),
        empty_label="Selecciona visitante",
        widget=forms.Select(attrs={"class": FORM_SELECT}),
    )
    fecha = forms.DateField(
        required=False,
        input_formats=DATE_INPUT_FORMATS,
        widget=forms.DateInput(
            format="%Y-%m-%d",
            attrs={"class": FORM_INPUT, "type": "date"},
        ),
    )
    goles_local = forms.IntegerField(
        required=False,
        min_value=0,
        widget=forms.NumberInput(attrs={"class": FORM_INPUT, "placeholder": "GL"}),
    )
    goles_visitante = forms.IntegerField(
        required=False,
        min_value=0,
        widget=forms.NumberInput(attrs={"class": FORM_INPUT, "placeholder": "GV"}),
    )

    def __init__(self, *args, temporada=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["local"].choices = obtener_equipos_choices(temporada)
        self.fields["visitante"].choices = obtener_equipos_choices(temporada)

    def clean_fecha(self):
        return parse_fecha_flexible(self.cleaned_data.get("fecha"))


def obtener_partidos_choices(temporada=None, jornada=None):
    """Devuelve los partidos disponibles para la Quiniela agrupados por división."""
    qs = Partido.objects.select_related("local", "visitante", "jornada__temporada").prefetch_related(
        "local__participaciones__division", "visitante__participaciones__division"
    )
    if jornada is not None:
        qs = qs.filter(jornada=jornada)
    elif temporada is not None:
        qs = qs.filter(jornada__temporada=temporada)

    partidos = list(qs.order_by("jornada__numero", "fecha", "id"))

    partidos_div1 = [p for p in partidos if p.division_nivel == 1]
    partidos_div2 = [p for p in partidos if p.division_nivel == 2]
    otros = [p for p in partidos if p.division_nivel not in (1, 2)]

    choices = [("", "— Selecciona partido —")]
    if partidos_div1:
        choices.append((
            "🏆 Primera División (LaLiga EA Sports)",
            [(p.id, f"{p.local.nombre} vs {p.visitante.nombre} (J{p.jornada.numero})") for p in partidos_div1],
        ))
    if partidos_div2:
        choices.append((
            "🥈 Segunda División (LaLiga Hypermotion)",
            [(p.id, f"{p.local.nombre} vs {p.visitante.nombre} (J{p.jornada.numero})") for p in partidos_div2],
        ))
    if otros:
        choices.append((
            "⚽ Otros Partidos",
            [(p.id, f"{p.local.nombre} vs {p.visitante.nombre} (J{p.jornada.numero})") for p in otros],
        ))
    return choices


class QuinielaForm(forms.ModelForm):
    class Meta:
        model = Quiniela
        fields = ["temporada", "jornada", "numero", "nombre", "n_dobles", "n_triples", "activa"]
        widgets = {
            "temporada": forms.Select(attrs={"class": FORM_SELECT}),
            "jornada": forms.Select(attrs={"class": FORM_SELECT}),
            "numero": forms.NumberInput(attrs={"class": FORM_INPUT, "placeholder": "Número de jornada"}),
            "nombre": forms.TextInput(attrs={"class": FORM_INPUT, "placeholder": "Ej: Quiniela Jornada 5"}),
            "n_dobles": forms.NumberInput(attrs={"class": FORM_INPUT, "min": 0, "max": 14}),
            "n_triples": forms.NumberInput(attrs={"class": FORM_INPUT, "min": 0, "max": 14}),
            "activa": forms.CheckboxInput(attrs={"class": FORM_CHECK}),
        }

    def __init__(self, *args, temporada=None, **kwargs):
        super().__init__(*args, **kwargs)
        temp = temporada or (self.instance.temporada if self.instance.pk else None) or Temporada.objects.filter(activa=True).first()
        if temp:
            self.fields["temporada"].initial = temp
            self.fields["jornada"].queryset = Jornada.objects.filter(temporada=temp).order_by("numero")
            if not self.instance.pk:
                ultima = Quiniela.objects.filter(temporada=temp).aggregate(Max("numero"))["numero__max"] or 0
                self.fields["numero"].initial = ultima + 1
                self.fields["nombre"].initial = f"Quiniela Jornada {ultima + 1}"
        self.fields["jornada"].required = False
        self.fields["jornada"].empty_label = "— Sin jornada fija (Partidos mixtos) —"