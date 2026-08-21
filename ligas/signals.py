"""Señales: mantienen la coherencia entre resultados reales y predicciones."""

from django.db.models.signals import post_save
from django.dispatch import receiver

from ligas.models import Partido, Prediccion


@receiver(post_save, sender=Partido)
def sincronizar_prediccion(sender, instance, **kwargs):
    """Si se corrige el resultado de un partido, actualiza su Prediccion."""
    prediccion = Prediccion.objects.filter(partido=instance).first()
    if prediccion is None:
        return
    if instance.jugado and instance.resultado is not None:
        prediccion.actualizar_con_resultado(instance.resultado)
    elif prediccion.resultado_real is not None or prediccion.acierto is not None:
        prediccion.resultado_real = None
        prediccion.acierto = None
        prediccion.save(update_fields=["resultado_real", "acierto"])