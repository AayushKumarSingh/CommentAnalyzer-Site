from django.apps import AppConfig
from .analyze import load_models


class AnalysisConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "analysis"


class LoadModels(AppConfig):
    name = 'analysis'

    def ready(self):
        load_models()

