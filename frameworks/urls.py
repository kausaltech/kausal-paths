from __future__ import annotations

from django.urls import path

from .views import create_result_file

urlpatterns = [
    path('fwc/<int:fwc_id>/<str:token>/result-file', create_result_file, name='framework_config_results_download')
]
